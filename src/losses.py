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
        losses = []

        for i in range(batch_size):
            pred_audio = pred[i].detach().cpu().numpy().T  # [samples, channels]
            target_audio = target[i].detach().cpu().numpy().T

            meter = pyln.Meter(self.sr)

            try:
                pred_lufs = meter.integrated_loudness(pred_audio)
                target_lufs = meter.integrated_loudness(target_audio)

                # Handle edge cases (inf or very low loudness)
                pred_is_valid = not (torch.isinf(torch.tensor(pred_lufs)) or pred_lufs < -70.0)
                target_is_valid = not (torch.isinf(torch.tensor(target_lufs)) or target_lufs < -70.0)

                if pred_is_valid and target_is_valid:
                    loss = abs(pred_lufs - target_lufs)
                    losses.append(torch.tensor(loss, device=pred.device, dtype=pred.dtype))
                else:
                    # Use RMS-based fallback for very quiet audio
                    pred_rms = torch.sqrt(torch.mean(pred[i] ** 2))
                    target_rms = torch.sqrt(torch.mean(target[i] ** 2))
                    rms_loss = torch.abs(pred_rms - target_rms) * 20.0  # Scale to ~LUFS range
                    losses.append(rms_loss)
            except Exception:
                # Fallback to RMS loss
                pred_rms = torch.sqrt(torch.mean(pred[i] ** 2))
                target_rms = torch.sqrt(torch.mean(target[i] ** 2))
                rms_loss = torch.abs(pred_rms - target_rms) * 20.0
                losses.append(rms_loss)

        result = torch.stack(losses).mean()
        # Clamp to prevent extreme values
        return torch.clamp(result, min=0.0, max=100.0)


class SpectralLoss(nn.Module):
    """
    Loss for matching log-mel spectrogram.

    Captures timbral and spectral characteristics.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
    ):
        """
        Initialize spectral loss.

        Args:
            n_mels: Number of mel frequency bins
            n_fft: FFT size
            hop_length: Hop length
            sample_rate: Audio sample rate
        """
        super().__init__()

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
        Compute spectral loss (L1 distance in log-mel space).

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]

        Returns:
            Mean L1 distance
        """
        # Move transforms to correct device if needed
        if self.mel_transform.mel_scale.fb.device != pred.device:
            self.mel_transform = self.mel_transform.to(pred.device)
            self.amplitude_to_db = self.amplitude_to_db.to(pred.device)

        # Convert to mono for spectral analysis
        if pred.shape[1] > 1:
            pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = target.mean(dim=1, keepdim=True)

        # Compute mel spectrograms
        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)

        # Convert to dB
        pred_log_mel = self.amplitude_to_db(pred_mel)
        target_log_mel = self.amplitude_to_db(target_mel)

        # L1 distance
        loss = F.l1_loss(pred_log_mel, target_log_mel)

        return loss


class DynamicRangeLoss(nn.Module):
    """
    Loss for matching dynamic range characteristics.

    Combines loudness range (LRA) and crest factor.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize dynamic range loss.

        Args:
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.sr = sample_rate

    def compute_lra(self, audio: torch.Tensor) -> float:
        """Compute loudness range."""
        audio_np = audio.detach().cpu().numpy().T
        meter = pyln.Meter(self.sr)

        try:
            # Compute short-term loudness blocks
            block_size = int(3.0 * self.sr)
            hop_size = block_size // 2

            loudness_blocks = []
            for i in range(0, len(audio_np) - block_size, hop_size):
                block = audio_np[i:i + block_size, :]
                try:
                    loudness = meter.integrated_loudness(block)
                    if not (torch.isinf(torch.tensor(loudness)) or torch.isnan(torch.tensor(loudness))):
                        loudness_blocks.append(loudness)
                except:
                    continue

            if len(loudness_blocks) < 2:
                return 0.0

            # LRA = 95th percentile - 10th percentile
            import numpy as np
            lra = np.percentile(loudness_blocks, 95) - np.percentile(loudness_blocks, 10)
            return max(0.0, float(lra))

        except Exception:
            return 0.0

    def compute_crest_factor(self, audio: torch.Tensor) -> float:
        """Compute crest factor (peak-to-RMS ratio) in dB."""
        peak = audio.abs().max().item()
        rms = (audio ** 2).mean().sqrt().item()

        if rms > 1e-10 and peak > 0:
            crest_db = 20 * torch.log10(torch.tensor(peak / rms)).item()
            return crest_db
        return 0.0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dynamic range loss.

        Args:
            pred: Predicted audio [batch, channels, samples]
            target: Target audio [batch, channels, samples]

        Returns:
            Combined LRA and crest factor loss
        """
        batch_size = pred.shape[0]
        lra_losses = []
        crest_losses = []

        for i in range(batch_size):
            # LRA loss
            pred_lra = self.compute_lra(pred[i])
            target_lra = self.compute_lra(target[i])
            lra_loss = abs(pred_lra - target_lra)
            lra_losses.append(torch.tensor(lra_loss, device=pred.device))

            # Crest factor loss
            pred_crest = self.compute_crest_factor(pred[i])
            target_crest = self.compute_crest_factor(target[i])
            crest_loss = abs(pred_crest - target_crest)
            crest_losses.append(torch.tensor(crest_loss, device=pred.device))

        lra_loss = torch.stack(lra_losses).mean()
        crest_loss = torch.stack(crest_losses).mean()

        # Combine with equal weight
        return (lra_loss + crest_loss) / 2.0


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for perceptual audio quality.

    Uses multiple FFT sizes to capture both fine and coarse spectral details.
    Reference: Yamamoto et al., "Parallel WaveGAN", 2020
    """

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_lengths: list = [512, 1024, 2048],
    ):
        """
        Initialize multi-resolution STFT loss.

        Args:
            fft_sizes: List of FFT sizes
            hop_sizes: List of hop lengths
            win_lengths: List of window lengths
        """
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

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
        # Convert to mono
        if pred.shape[1] > 1:
            pred = pred.mean(dim=1)
        else:
            pred = pred.squeeze(1)

        if target.shape[1] > 1:
            target = target.mean(dim=1)
        else:
            target = target.squeeze(1)

        total_loss = 0.0

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFTs
            pred_stft = self.stft(pred, fft_size, hop_size, win_length)
            target_stft = self.stft(target, fft_size, hop_size, win_length)

            # Magnitude
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            # Spectral convergence loss
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / torch.norm(target_mag, p="fro")

            # Log magnitude loss
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-5),
                torch.log(target_mag + 1e-5),
            )

            total_loss += (sc_loss + log_mag_loss)

        return total_loss / len(self.fft_sizes)


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
        # Compute individual losses
        loss_loudness = self.loudness_loss(pred, target)
        loss_spectral = self.spectral_loss(pred, target)
        loss_dynamic = self.dynamic_loss(pred, target)
        loss_perceptual = self.perceptual_loss(pred, target)

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
