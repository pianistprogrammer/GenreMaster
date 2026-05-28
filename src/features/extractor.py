"""Audio feature extraction for mastering analysis.

Extracts mastering-relevant features:
- LUFS (Loudness Units relative to Full Scale) - integrated loudness
- LRA (Loudness Range) - dynamic range in LU
- True Peak (dBTP) - maximum true peak level
- Spectral Tilt - slope of log-magnitude spectrum
- M-S Ratio - mid-side stereo width metric
- DR14 - dynamic range metric (14-bit scale)
"""

from typing import Dict, Optional

import numpy as np
import torch
import pyloudnorm as pyln
from scipy import signal
from scipy.stats import linregress


class AudioFeatureExtractor:
    """Extract mastering-relevant audio features."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Audio sample rate (default: 44100 Hz)
        """
        self.sr = sample_rate
        self.meter = pyln.Meter(sample_rate)  # ITU-R BS.1770-4 loudness meter

    def extract_lufs(self, waveform: torch.Tensor) -> float:
        """
        Extract integrated LUFS (Loudness Units relative to Full Scale).

        Uses ITU-R BS.1770-4 standard for loudness measurement.

        Args:
            waveform: Audio tensor of shape [channels, samples]

        Returns:
            Integrated LUFS value in dB (typically -30 to 0)
        """
        # Convert to numpy and ensure proper shape for pyloudnorm
        audio_np = waveform.cpu().numpy()

        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        # pyloudnorm expects shape (samples, channels), so transpose
        audio_np = audio_np.T

        try:
            loudness = self.meter.integrated_loudness(audio_np)
            # Handle edge case: silent audio returns -inf
            if np.isinf(loudness) or np.isnan(loudness):
                return -70.0  # Minimum representable loudness
            return float(loudness)
        except Exception as e:
            # Fallback for problematic audio
            return -70.0

    def extract_lra(self, waveform: torch.Tensor) -> float:
        """
        Extract LRA (Loudness Range) in Loudness Units.

        Measures the variation in loudness over time (dynamic range).

        Args:
            waveform: Audio tensor of shape [channels, samples]

        Returns:
            LRA value in LU (typically 2-25)
        """
        audio_np = waveform.cpu().numpy()

        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        # Transpose for pyloudnorm
        audio_np = audio_np.T

        try:
            # Compute short-term loudness (3s gating)
            # LRA = difference between 95th and 10th percentile of gated loudness
            block_size = int(3.0 * self.sr)  # 3-second blocks
            hop_size = block_size // 2

            loudness_blocks = []
            for i in range(0, len(audio_np) - block_size, hop_size):
                block = audio_np[i:i + block_size, :]
                try:
                    block_loudness = self.meter.integrated_loudness(block)
                    if not (np.isinf(block_loudness) or np.isnan(block_loudness)):
                        loudness_blocks.append(block_loudness)
                except:
                    continue

            if len(loudness_blocks) < 2:
                return 0.0  # Insufficient data

            # LRA = 95th percentile - 10th percentile
            lra = np.percentile(loudness_blocks, 95) - np.percentile(loudness_blocks, 10)
            return float(max(0.0, lra))

        except Exception:
            return 0.0

    def extract_true_peak(self, waveform: torch.Tensor) -> float:
        """
        Extract true peak level in dBTP (decibels True Peak).

        Uses 4x oversampling to detect inter-sample peaks.

        Args:
            waveform: Audio tensor of shape [channels, samples]

        Returns:
            True peak level in dBTP (typically -10 to 0)
        """
        audio_np = waveform.cpu().numpy()

        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        # Upsample by 4x to detect inter-sample peaks
        upsampled = signal.resample(audio_np, audio_np.shape[1] * 4, axis=1)

        # Find maximum absolute value across all channels
        max_peak = np.max(np.abs(upsampled))

        # Convert to dB
        if max_peak > 0:
            true_peak_db = 20 * np.log10(max_peak)
        else:
            true_peak_db = -100.0  # Silence

        return float(true_peak_db)

    def extract_spectral_tilt(
        self,
        waveform: torch.Tensor,
        n_fft: int = 2048,
        freq_range: tuple = (100, 10000),
    ) -> float:
        """
        Extract spectral tilt (slope of log-magnitude spectrum).

        Measures overall frequency balance: positive = bright, negative = dark.

        Args:
            waveform: Audio tensor of shape [channels, samples]
            n_fft: FFT size
            freq_range: Frequency range for linear regression (Hz)

        Returns:
            Spectral tilt in dB/octave
        """
        audio_np = waveform.cpu().numpy()

        # Convert to mono for spectral analysis
        if audio_np.ndim == 2 and audio_np.shape[0] > 1:
            audio_np = np.mean(audio_np, axis=0)
        elif audio_np.ndim == 2:
            audio_np = audio_np[0, :]

        # Compute magnitude spectrum
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sr)
        spectrum = np.fft.rfft(audio_np, n=n_fft)
        magnitude = np.abs(spectrum)

        # Apply frequency range filter
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_filtered = freqs[freq_mask]
        magnitude_filtered = magnitude[freq_mask]

        # Convert to log scale
        log_freqs = np.log10(freqs_filtered + 1e-10)
        log_magnitude = 20 * np.log10(magnitude_filtered + 1e-10)

        # Linear regression: slope indicates tilt
        try:
            slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_magnitude)
            # Convert slope to dB/octave
            tilt_db_per_octave = slope * np.log10(2)
            return float(tilt_db_per_octave)
        except Exception:
            return 0.0

    def extract_ms_ratio(self, waveform: torch.Tensor) -> float:
        """
        Extract mid-side ratio (stereo width metric).

        M-S ratio = RMS(Side) / RMS(Mid)
        Higher values indicate wider stereo image.

        Args:
            waveform: Audio tensor of shape [2, samples] (stereo required)

        Returns:
            M-S ratio (0 = mono, >1 = wide stereo)
        """
        audio_np = waveform.cpu().numpy()

        # Require stereo input
        if audio_np.ndim == 1 or audio_np.shape[0] == 1:
            return 0.0  # Mono signal

        # Ensure exactly 2 channels
        left = audio_np[0, :]
        right = audio_np[1, :] if audio_np.shape[0] > 1 else audio_np[0, :]

        # Convert to mid-side
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        # Compute RMS
        rms_mid = np.sqrt(np.mean(mid ** 2))
        rms_side = np.sqrt(np.mean(side ** 2))

        # M-S ratio
        if rms_mid > 1e-10:
            ms_ratio = rms_side / rms_mid
        else:
            ms_ratio = 0.0

        return float(ms_ratio)

    def extract_dr14(self, waveform: torch.Tensor, block_duration: float = 3.0) -> float:
        """
        Extract DR14 dynamic range metric.

        DR14 = dB difference between peak and RMS of the loudest 20% of blocks.

        Args:
            waveform: Audio tensor of shape [channels, samples]
            block_duration: Block size in seconds (default: 3.0)

        Returns:
            DR14 value in dB (typically 5-20)
        """
        audio_np = waveform.cpu().numpy()

        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        # Convert to mono for DR measurement
        if audio_np.shape[0] > 1:
            audio_mono = np.mean(audio_np, axis=0)
        else:
            audio_mono = audio_np[0, :]

        # Split into blocks
        block_size = int(block_duration * self.sr)
        n_blocks = len(audio_mono) // block_size

        if n_blocks < 2:
            return 0.0

        block_rms = []
        for i in range(n_blocks):
            block = audio_mono[i * block_size:(i + 1) * block_size]
            rms = np.sqrt(np.mean(block ** 2))
            if rms > 1e-10:
                block_rms.append(rms)

        if len(block_rms) < 2:
            return 0.0

        # Sort blocks by RMS
        block_rms_sorted = np.sort(block_rms)[::-1]  # Descending

        # Take loudest 20% of blocks
        n_top = max(1, int(0.2 * len(block_rms_sorted)))
        top_blocks = block_rms_sorted[:n_top]

        # Compute average RMS of top blocks
        avg_rms = np.mean(top_blocks)

        # Compute peak level
        peak = np.max(np.abs(audio_mono))

        # DR14 = peak_dB - rms_dB
        if peak > 0 and avg_rms > 0:
            peak_db = 20 * np.log10(peak)
            rms_db = 20 * np.log10(avg_rms)
            dr14 = peak_db - rms_db
            return float(max(0.0, dr14))
        else:
            return 0.0

    def extract_all_features(self, waveform: torch.Tensor) -> Dict[str, float]:
        """
        Extract all mastering features from audio.

        Args:
            waveform: Audio tensor of shape [channels, samples]

        Returns:
            Dictionary of feature names and values
        """
        features = {
            'lufs': self.extract_lufs(waveform),
            'lra': self.extract_lra(waveform),
            'true_peak': self.extract_true_peak(waveform),
            'spectral_tilt': self.extract_spectral_tilt(waveform),
            'ms_ratio': self.extract_ms_ratio(waveform),
            'dr14': self.extract_dr14(waveform),
        }

        return features


def extract_features_from_file(audio_path: str, sample_rate: int = 44100) -> Dict[str, float]:
    """
    Convenience function to extract features directly from audio file.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Dictionary of features
    """
    from utils import load_audio

    waveform, sr = load_audio(audio_path, sr=sample_rate, mono=False)
    extractor = AudioFeatureExtractor(sample_rate=sr)
    return extractor.extract_all_features(waveform)
