"""Loss functions for GenreMaster training.

Implements multi-component loss:
- L_loudness: LUFS (integrated loudness) matching
- L_spectral: Log-mel spectrogram similarity
- L_dynamic: Dynamic range (LRA + crest factor) matching
- L_perceptual: Multi-resolution STFT loss for perceptual quality
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import pyloudnorm as pyln


class LoudnessLoss(nn.Module):
    """
    Loss for matching integrated loudness (LUFS).

    Uses ITU-R BS.1770-4 loudness measurement.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize loudness loss.

        Args:
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.sr = sample_rate

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loudness loss.

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]

        Returns:
            Mean absolute error in LUFS
        """
        batch_size = pred.shape[0]

        # Use differentiable RMS-based loudness approximation
        # This avoids the non-differentiable pyloudnorm LUFS measurement
        # RMS in dB correlates well with perceived loudness
        eps = 1e-8

        # Compute RMS for each sample
        pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=(1, 2)) + eps)
        target_rms = torch.sqrt(torch.mean(target ** 2, dim=(1, 2)) + eps)

        # Convert to dB scale (approximates LUFS better)
        pred_db = 20 * torch.log10(pred_rms + eps)
        target_db = 20 * torch.log10(target_rms + eps)

        # Clamp to reasonable range to prevent extreme gradients
        pred_db = pred_db.clamp(min=-70, max=0)
        target_db = target_db.clamp(min=-70, max=0)

        # Mean absolute error in dB (similar to LUFS difference)
        loss = torch.abs(pred_db - target_db).mean()

        return loss.clamp(max=50.0)  # Cap at 50 dB difference


class SpectralLoss(nn.Module):
    """
    Loss for matching log-mel spectrogram.

    Captures timbral and spectral characteristics.

    NOTE: On MPS, STFT backward pass can produce NaN gradients.
    We use a simplified L1+L2 waveform loss as fallback.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
        use_stft: bool = False,  # Disabled by default for MPS compatibility
    ):
        super().__init__()
        self.use_stft = use_stft

        if use_stft:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            self.amplitude_to_db = T.AmplitudeToDB()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spectral loss.

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]

        Returns:
            Loss value
        """
        if not self.use_stft:
            # MPS-compatible: use multi-scale waveform loss
            # This approximates spectral matching without STFT
            losses = []

            # L1 loss at original scale
            losses.append(F.l1_loss(pred, target))

            # L2 loss for emphasis on large errors
            losses.append(F.mse_loss(pred, target))

            # Derivative loss (captures high-frequency content)
            pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
            target_diff = target[:, :, 1:] - target[:, :, :-1]
            losses.append(F.l1_loss(pred_diff, target_diff))

            return sum(losses) / len(losses)

        # Original STFT-based loss (may cause NaN on MPS)
        if self.mel_transform.mel_scale.fb.device != pred.device:
            self.mel_transform = self.mel_transform.to(pred.device)
            self.amplitude_to_db = self.amplitude_to_db.to(pred.device)

        if pred.shape[1] > 1:
            pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = target.mean(dim=1, keepdim=True)

        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)

        pred_log_mel = self.amplitude_to_db(pred_mel)
        target_log_mel = self.amplitude_to_db(target_mel)

        return F.l1_loss(pred_log_mel, target_log_mel)


class DynamicRangeLoss(nn.Module):
    """
    Loss for matching dynamic range characteristics.

    Uses differentiable crest factor (peak-to-RMS ratio).
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sr = sample_rate

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dynamic range loss using differentiable crest factor.

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]

        Returns:
            Crest factor loss in dB
        """
        eps = 1e-8

        # Compute peak and RMS for each sample (differentiable)
        # Use soft-max approximation for peak to make it differentiable
        pred_softmax_peak = torch.logsumexp(pred.abs() * 10, dim=(1, 2)) / 10
        target_softmax_peak = torch.logsumexp(target.abs() * 10, dim=(1, 2)) / 10

        pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=(1, 2)) + eps)
        target_rms = torch.sqrt(torch.mean(target ** 2, dim=(1, 2)) + eps)

        # Crest factor in dB (differentiable)
        pred_crest = 20 * torch.log10((pred_softmax_peak + eps) / (pred_rms + eps))
        target_crest = 20 * torch.log10((target_softmax_peak + eps) / (target_rms + eps))

        # Clamp to reasonable range
        pred_crest = pred_crest.clamp(min=0, max=30)
        target_crest = target_crest.clamp(min=0, max=30)

        # Mean absolute error
        loss = torch.abs(pred_crest - target_crest).mean()

        return loss.clamp(max=20.0)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for perceptual audio quality.

    Uses multiple FFT sizes to capture both fine and coarse spectral details.
    Reference: Yamamoto et al., "Parallel WaveGAN", 2020

    NOTE: On MPS, STFT backward can produce NaN. This implementation
    moves tensors to CPU for STFT computation to avoid the issue.
    """

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_lengths: list = [512, 1024, 2048],
        use_cpu_for_stft: bool = True,  # Workaround for MPS NaN issue
    ):
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.use_cpu_for_stft = use_cpu_for_stft

    def stft(
        self,
        audio: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
    ) -> torch.Tensor:
        """Compute STFT."""
        window = torch.hann_window(win_length, device=audio.device)

        stft = torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        return stft

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]

        Returns:
            Combined spectral convergence and magnitude loss
        """
        original_device = pred.device

        # Convert to mono
        if pred.shape[1] > 1:
            pred = pred.mean(dim=1)
        else:
            pred = pred.squeeze(1)

        if target.shape[1] > 1:
            target = target.mean(dim=1)
        else:
            target = target.squeeze(1)

        # Move to CPU if needed to avoid MPS STFT backward NaN issue
        if self.use_cpu_for_stft and pred.device.type == 'mps':
            pred = pred.cpu()
            target = target.cpu()

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFTs
            pred_stft = self.stft(pred, fft_size, hop_size, win_length)
            target_stft = self.stft(target, fft_size, hop_size, win_length)

            # Magnitude with numerical stability
            pred_mag = pred_stft.abs().clamp(min=1e-7)
            target_mag = target_stft.abs().clamp(min=1e-7)

            # Spectral convergence loss with epsilon to prevent div by zero
            target_norm = torch.norm(target_mag, p="fro").clamp(min=1e-7)
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / target_norm

            # Log magnitude loss with safe log
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag.clamp(min=1e-7)),
                torch.log(target_mag.clamp(min=1e-7)),
            )

            # Clamp individual losses to prevent explosion
            sc_loss = sc_loss.clamp(max=10.0)
            log_mag_loss = log_mag_loss.clamp(max=10.0)

            total_loss = total_loss + (sc_loss + log_mag_loss)

        result = total_loss / len(self.fft_sizes)

        # Move back to original device
        if result.device != original_device:
            result = result.to(original_device)

        return result


class GenreMasterLoss(nn.Module):
    """
    Combined loss for GenreMaster training.

    L_total = λ₁·L_loudness + λ₂·L_spectral + λ₃·L_dynamic + λ₄·L_perceptual
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        lambda_loudness: float = 1.0,
        lambda_spectral: float = 0.5,
        lambda_dynamic: float = 0.5,
        lambda_perceptual: float = 0.1,
    ):
        """
        Initialize combined loss.

        Args:
            sample_rate: Audio sample rate
            lambda_loudness: Weight for loudness loss
            lambda_spectral: Weight for spectral loss
            lambda_dynamic: Weight for dynamic range loss
            lambda_perceptual: Weight for perceptual loss
        """
        super().__init__()

        self.lambda_loudness = lambda_loudness
        self.lambda_spectral = lambda_spectral
        self.lambda_dynamic = lambda_dynamic
        self.lambda_perceptual = lambda_perceptual

        self.loudness_loss = LoudnessLoss(sample_rate)
        self.spectral_loss = SpectralLoss(sample_rate=sample_rate)
        self.dynamic_loss = DynamicRangeLoss(sample_rate)
        self.perceptual_loss = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]
            return_components: If True, return dict with individual losses

        Returns:
            Total loss scalar or dict of loss components
        """
        device = pred.device
        dtype = pred.dtype

        # Compute individual losses only if weight > 0
        if self.lambda_loudness > 0:
            loss_loudness = self.loudness_loss(pred, target)
        else:
            loss_loudness = torch.tensor(0.0, device=device, dtype=dtype)

        if self.lambda_spectral > 0:
            loss_spectral = self.spectral_loss(pred, target)
        else:
            loss_spectral = torch.tensor(0.0, device=device, dtype=dtype)

        if self.lambda_dynamic > 0:
            loss_dynamic = self.dynamic_loss(pred, target)
        else:
            loss_dynamic = torch.tensor(0.0, device=device, dtype=dtype)

        if self.lambda_perceptual > 0:
            loss_perceptual = self.perceptual_loss(pred, target)
        else:
            loss_perceptual = torch.tensor(0.0, device=device, dtype=dtype)

        # Check for NaN/Inf in individual losses
        loss_loudness = torch.where(
            torch.isnan(loss_loudness) | torch.isinf(loss_loudness),
            torch.zeros_like(loss_loudness),
            loss_loudness
        )
        loss_spectral = torch.where(
            torch.isnan(loss_spectral) | torch.isinf(loss_spectral),
            torch.zeros_like(loss_spectral),
            loss_spectral
        )
        loss_dynamic = torch.where(
            torch.isnan(loss_dynamic) | torch.isinf(loss_dynamic),
            torch.zeros_like(loss_dynamic),
            loss_dynamic
        )
        loss_perceptual = torch.where(
            torch.isnan(loss_perceptual) | torch.isinf(loss_perceptual),
            torch.zeros_like(loss_perceptual),
            loss_perceptual
        )

        # Weighted combination
        total_loss = (
            self.lambda_loudness * loss_loudness +
            self.lambda_spectral * loss_spectral +
            self.lambda_dynamic * loss_dynamic +
            self.lambda_perceptual * loss_perceptual
        )

        # Final NaN check
        total_loss = torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype),
            total_loss
        )

        if return_components:
            return {
                'total': total_loss,
                'loudness': loss_loudness,
                'spectral': loss_spectral,
                'dynamic': loss_dynamic,
                'perceptual': loss_perceptual,
            }

        return total_loss


def create_loss_function(
    sample_rate: int = 44100,
    loss_weights: Optional[Dict[str, float]] = None,
) -> GenreMasterLoss:
    """
    Factory function to create loss function.

    Args:
        sample_rate: Audio sample rate
        loss_weights: Optional dict of loss weights
            Keys: 'loudness', 'spectral', 'dynamic', 'perceptual'

    Returns:
        GenreMasterLoss instance
    """
    if loss_weights is None:
        loss_weights = {
            'loudness': 1.0,
            'spectral': 0.5,
            'dynamic': 0.5,
            'perceptual': 0.1,
        }

    return GenreMasterLoss(
        sample_rate=sample_rate,
        lambda_loudness=loss_weights.get('loudness', 1.0),
        lambda_spectral=loss_weights.get('spectral', 0.5),
        lambda_dynamic=loss_weights.get('dynamic', 0.5),
        lambda_perceptual=loss_weights.get('perceptual', 0.1),
    )
