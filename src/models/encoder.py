"""Spectral encoder for audio feature extraction.

Implements ResNet-based encoder that processes log-mel spectrograms
to extract audio features for conditioning the mastering chain.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpectralEncoder(nn.Module):
    """
    ResNet-based spectral encoder for audio analysis.

    Processes log-mel spectrograms to extract semantic audio features.
    Architecture: Log-Mel → ResNet-18 → Global Pool → Feature Vector
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
        feature_dim: int = 512,
        pretrained: bool = False,
    ):
        """
        Initialize spectral encoder.

        Args:
            n_mels: Number of mel frequency bins
            n_fft: FFT size
            hop_length: Hop length for STFT
            sample_rate: Audio sample rate
            feature_dim: Output feature dimension
            pretrained: Use pretrained ResNet weights (transfer learning)
        """
        super().__init__()

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sample_rate
        self.feature_dim = feature_dim

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB()

        # ResNet-18 backbone (modified for single-channel input)
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        # Modify first conv layer for single-channel spectrogram input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final classification layer
        self.backbone.fc = nn.Identity()

        # Feature projection to desired dimension
        self.feature_proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram from waveform.

        Args:
            waveform: Audio tensor of shape [batch, channels, samples]

        Returns:
            Log-mel spectrogram of shape [batch, 1, n_mels, time]
        """
        # Convert to mono if stereo
        if waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # Add channel dim

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to dB scale
        log_mel = self.amplitude_to_db(mel_spec)

        # Normalize to [-1, 1] range for better training
        log_mel = (log_mel + 80) / 80  # Assume typical range [-80, 0] dB

        return log_mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: waveform → features.

        Args:
            waveform: Audio tensor of shape [batch, channels, samples]

        Returns:
            Feature tensor of shape [batch, feature_dim]
        """
        # Compute log-mel spectrogram
        log_mel = self.compute_spectrogram(waveform)

        # Pass through ResNet backbone
        features = self.backbone(log_mel)

        # Project to desired dimension
        features = self.feature_proj(features)

        return features


class LightweightSpectralEncoder(nn.Module):
    """
    Lightweight CNN-based spectral encoder (faster alternative to ResNet).

    Uses a simpler architecture for faster training and inference.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
        feature_dim: int = 512,
    ):
        """
        Initialize lightweight encoder.

        Args:
            n_mels: Number of mel frequency bins
            n_fft: FFT size
            hop_length: Hop length for STFT
            sample_rate: Audio sample rate
            feature_dim: Output feature dimension
        """
        super().__init__()

        self.n_mels = n_mels
        self.feature_dim = feature_dim

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.amplitude_to_db = T.AmplitudeToDB()

        # Lightweight CNN architecture
        self.conv_layers = nn.Sequential(
            # Layer 1: [1, 128, T] → [64, 64, T/2]
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: [64, 64, T/2] → [128, 32, T/4]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: [128, 32, T/4] → [256, 16, T/8]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4: [256, 16, T/8] → [512, 8, T/16]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram."""
        # Convert to mono if stereo
        if waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to dB scale and normalize
        log_mel = self.amplitude_to_db(mel_spec)
        log_mel = (log_mel + 80) / 80

        return log_mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: waveform → features.

        Args:
            waveform: Audio tensor of shape [batch, channels, samples]

        Returns:
            Feature tensor of shape [batch, feature_dim]
        """
        # Compute log-mel spectrogram
        log_mel = self.compute_spectrogram(waveform)

        # Pass through CNN
        features = self.conv_layers(log_mel)

        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)

        # Project to desired dimension
        features = self.feature_proj(features)

        return features


def create_encoder(
    encoder_type: str = "lightweight",
    feature_dim: int = 512,
    sample_rate: int = 44100,
    pretrained: bool = False,
) -> nn.Module:
    """
    Factory function to create spectral encoder.

    Args:
        encoder_type: 'resnet' or 'lightweight'
        feature_dim: Output feature dimension
        sample_rate: Audio sample rate
        pretrained: Use pretrained weights (ResNet only)

    Returns:
        Spectral encoder module
    """
    if encoder_type == "resnet":
        return SpectralEncoder(
            feature_dim=feature_dim,
            sample_rate=sample_rate,
            pretrained=pretrained,
        )
    elif encoder_type == "lightweight":
        return LightweightSpectralEncoder(
            feature_dim=feature_dim,
            sample_rate=sample_rate,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
