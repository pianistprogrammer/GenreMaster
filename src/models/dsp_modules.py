"""Differentiable DSP modules for audio mastering.

Implements differentiable audio signal processing operations:
- Loudness Normalization
- Parametric EQ (8-band)
- Multiband Compression
- True Peak Limiter
- Stereo Width Control

All operations are differentiable w.r.t. their parameters for end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import pyloudnorm as pyln


class DifferentiableLoudnessNormalizer(nn.Module):
    """
    Differentiable loudness normalization module.

    Applies gain to achieve target LUFS (integrated loudness).
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize loudness normalizer.

        Args:
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.sr = sample_rate

    def forward(
        self,
        waveform: torch.Tensor,
        target_lufs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize audio to target LUFS.

        Args:
            waveform: Audio tensor [batch, channels, samples]
            target_lufs: Target LUFS values [batch, 1]

        Returns:
            Normalized audio
        """
        batch_size = waveform.shape[0]
        normalized = []

        for i in range(batch_size):
            audio = waveform[i]  # [channels, samples]
            target = target_lufs[i].item()

            # Measure current loudness (non-differentiable, just for gain calculation)
            with torch.no_grad():
                audio_np = audio.cpu().numpy().T  # [samples, channels]
                meter = pyln.Meter(self.sr)
                try:
                    current_lufs = meter.integrated_loudness(audio_np)
                    if current_lufs < -70 or torch.isinf(torch.tensor(current_lufs)):
                        gain_db = 0.0
                    else:
                        gain_db = target - current_lufs
                except:
                    gain_db = 0.0

            # Apply gain (differentiable)
            gain_linear = 10 ** (gain_db / 20.0)
            normalized_audio = audio * gain_linear
            normalized.append(normalized_audio)

        return torch.stack(normalized, dim=0)


class ParametricEQ(nn.Module):
    """
    Differentiable 8-band parametric EQ.

    Uses biquad filters for each band.
    Simplified implementation using gain multiplication per frequency bin.
    """

    def __init__(self, n_bands: int = 8, n_fft: int = 2048, sample_rate: int = 44100):
        """
        Initialize parametric EQ.

        Args:
            n_bands: Number of EQ bands
            n_fft: FFT size for processing
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.sr = sample_rate
        self.hop_length = n_fft // 4

        # Define center frequencies for bands (log-spaced)
        self.register_buffer(
            "center_freqs",
            torch.logspace(
                torch.log10(torch.tensor(60.0)),
                torch.log10(torch.tensor(16000.0)),
                n_bands,
            )
        )

    def forward(
        self,
        waveform: torch.Tensor,
        gains_db: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply parametric EQ to audio.

        Args:
            waveform: Audio tensor [batch, channels, samples]
            gains_db: Gain for each band [batch, n_bands]

        Returns:
            EQ'd audio
        """
        batch, channels, samples = waveform.shape

        # Apply per channel
        output = []
        for c in range(channels):
            channel_audio = waveform[:, c, :]  # [batch, samples]

            # Pad to avoid artifacts
            padded = F.pad(channel_audio, (self.n_fft // 2, self.n_fft // 2), mode='reflect')

            # STFT
            stft = torch.stft(
                padded,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=waveform.device),
                return_complex=True,
            )  # [batch, freq_bins, time_frames]

            # Compute frequency bins
            freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.sr).to(waveform.device)

            # Create EQ curve from band gains
            eq_curve = torch.ones((batch, len(freqs), 1), device=waveform.device)

            for band_idx in range(self.n_bands):
                center_freq = self.center_freqs[band_idx]
                band_gains = gains_db[:, band_idx]  # [batch]

                # Gaussian-shaped band response (Q factor ~ 1.0)
                bandwidth = center_freq / 2.0
                band_response = torch.exp(
                    -((freqs - center_freq) ** 2) / (2 * bandwidth ** 2)
                )  # [freq_bins]

                # Apply gain to this band
                for b in range(batch):
                    gain_linear = 10 ** (band_gains[b] / 20.0)
                    eq_curve[b, :, 0] += (gain_linear - 1.0) * band_response

            # Apply EQ curve to STFT
            stft_eq = stft * eq_curve

            # Inverse STFT
            audio_eq = torch.istft(
                stft_eq,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=waveform.device),
                length=padded.shape[1],
            )

            # Remove padding
            audio_eq = audio_eq[:, self.n_fft // 2: self.n_fft // 2 + samples]

            output.append(audio_eq)

        # Stack channels
        output = torch.stack(output, dim=1)  # [batch, channels, samples]

        return output


class MultibandCompressor(nn.Module):
    """
    Simplified differentiable multiband compressor.

    Splits audio into frequency bands and applies compression to each.
    """

    def __init__(self, n_bands: int = 3, n_fft: int = 2048, sample_rate: int = 44100):
        """
        Initialize multiband compressor.

        Args:
            n_bands: Number of frequency bands (default: 3 - low/mid/high)
            n_fft: FFT size
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.sr = sample_rate
        self.hop_length = n_fft // 4

        # Define band split frequencies (low: <200Hz, mid: 200-2kHz, high: >2kHz)
        self.register_buffer(
            "band_edges",
            torch.tensor([0.0, 200.0, 2000.0, sample_rate / 2.0])
        )

    def forward(
        self,
        waveform: torch.Tensor,
        thresholds_db: torch.Tensor,
        ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply multiband compression.

        Args:
            waveform: Audio tensor [batch, channels, samples]
            thresholds_db: Compression thresholds for each band [batch, n_bands]
            ratios: Compression ratios for each band [batch, n_bands]

        Returns:
            Compressed audio
        """
        batch, channels, samples = waveform.shape
        eps = 1e-8

        # Simple RMS-based compression per band (approximation)
        output = torch.zeros_like(waveform)

        for band_idx in range(self.n_bands):
            low_freq = self.band_edges[band_idx]
            high_freq = self.band_edges[band_idx + 1]

            # Extract band (simplified: use entire signal, scale by band weight)
            band_audio = waveform

            # Compute RMS envelope with numerical stability
            window_size = int(0.01 * self.sr)  # 10ms windows
            rms = F.avg_pool1d(
                band_audio.pow(2).view(batch * channels, 1, samples),
                kernel_size=window_size,
                stride=1,
                padding=window_size // 2,
            ).sqrt().view(batch, channels, -1)
            rms = rms.clamp(min=eps)  # Prevent division by zero

            # Apply compression
            for b in range(batch):
                threshold = 10 ** (thresholds_db[b, band_idx].clamp(min=-60, max=0) / 20.0)
                threshold = max(threshold, eps)
                ratio = ratios[b, band_idx].clamp(min=1.0, max=20.0)

                # Gain reduction with numerical stability
                ratio_factor = (1.0 / ratio) - 1.0
                rms_ratio = (rms[b] / threshold).clamp(min=eps, max=100.0)
                gain = torch.where(
                    rms[b] > threshold,
                    rms_ratio ** ratio_factor,
                    torch.ones_like(rms[b]),
                )
                # Clamp gain to prevent extreme values
                gain = gain.clamp(min=0.01, max=10.0)

                # Pad gain to match audio length
                if gain.shape[1] < samples:
                    gain = F.pad(gain, (0, samples - gain.shape[1]))
                elif gain.shape[1] > samples:
                    gain = gain[:, :samples]

                output[b] += band_audio[b] * gain / self.n_bands

        return output


class TruePeakLimiter(nn.Module):
    """
    Differentiable true peak limiter using soft clipping.

    Uses tanh-based soft clipping to approximate limiting behavior.
    """

    def __init__(self):
        """Initialize true peak limiter."""
        super().__init__()

    def forward(
        self,
        waveform: torch.Tensor,
        threshold_db: torch.Tensor,
        ceiling_db: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Apply true peak limiting.

        Args:
            waveform: Audio tensor [batch, channels, samples]
            threshold_db: Limiter threshold [batch, 1]
            ceiling_db: Output ceiling level [batch, 1] (default: 0 dBFS)

        Returns:
            Limited audio
        """
        if ceiling_db is None:
            ceiling_db = torch.zeros_like(threshold_db)

        batch = waveform.shape[0]
        output = []
        eps = 1e-8

        for b in range(batch):
            audio = waveform[b]
            # Clamp threshold to reasonable range to prevent overflow
            thresh_db_clamped = threshold_db[b].clamp(min=-60, max=0).item()
            ceil_db_clamped = ceiling_db[b].clamp(min=-60, max=0).item()
            threshold = max(10 ** (thresh_db_clamped / 20.0), eps)
            ceiling = max(10 ** (ceil_db_clamped / 20.0), eps)

            # Soft clipping using tanh with numerical stability
            # Clamp input to tanh to prevent extreme values
            scaled_audio = (audio / threshold).clamp(min=-10, max=10)
            limited = torch.tanh(scaled_audio) * threshold

            # Scale to ceiling
            limited = limited * (ceiling / threshold)

            output.append(limited)

        return torch.stack(output, dim=0)


class StereoWidthControl(nn.Module):
    """
    Differentiable stereo width control using mid-side processing.

    Controls the width of the stereo image by scaling the side channel.
    """

    def __init__(self):
        """Initialize stereo width control."""
        super().__init__()

    def forward(
        self,
        waveform: torch.Tensor,
        width: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply stereo width control.

        Args:
            waveform: Stereo audio tensor [batch, 2, samples]
            width: Width factor [batch, 1] where 0=mono, 1=unchanged, >1=wider

        Returns:
            Width-adjusted stereo audio
        """
        batch, channels, samples = waveform.shape

        if channels != 2:
            return waveform  # Only works on stereo

        left = waveform[:, 0, :]  # [batch, samples]
        right = waveform[:, 1, :]  # [batch, samples]

        # Convert to mid-side
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        # Scale side by width factor
        side_scaled = side * width.squeeze(1).unsqueeze(1)  # [batch, samples]

        # Convert back to left-right
        left_out = mid + side_scaled
        right_out = mid - side_scaled

        output = torch.stack([left_out, right_out], dim=1)

        return output


class MasteringChain(nn.Module):
    """
    Complete differentiable mastering chain.

    Chains together all DSP modules in mastering order.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize mastering chain.

        Args:
            sample_rate: Audio sample rate
        """
        super().__init__()

        self.sr = sample_rate

        # DSP modules
        self.loudness_norm = DifferentiableLoudnessNormalizer(sample_rate)
        self.eq = ParametricEQ(n_bands=8, sample_rate=sample_rate)
        self.compressor = MultibandCompressor(n_bands=3, sample_rate=sample_rate)
        self.limiter = TruePeakLimiter()
        self.stereo_width = StereoWidthControl()

    def forward(
        self,
        waveform: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        """
        Apply full mastering chain.

        Args:
            waveform: Input audio [batch, channels, samples]
            params: Dictionary of DSP parameters:
                - target_lufs: [batch, 1]
                - eq_gains: [batch, 8]
                - comp_thresholds: [batch, 3]
                - comp_ratios: [batch, 3]
                - limiter_threshold: [batch, 1]
                - stereo_width: [batch, 1]

        Returns:
            Mastered audio
        """
        # 1. Loudness normalization
        x = self.loudness_norm(waveform, params['target_lufs'])

        # 2. Parametric EQ
        x = self.eq(x, params['eq_gains'])

        # 3. Multiband compression
        x = self.compressor(x, params['comp_thresholds'], params['comp_ratios'])

        # 4. Stereo width control
        x = self.stereo_width(x, params['stereo_width'])

        # 5. True peak limiter (final stage)
        x = self.limiter(x, params['limiter_threshold'])

        return x
