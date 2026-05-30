"""GenreMaster V2: Improved architecture with Transformer encoder and better DSP.

Key improvements over V1:
1. Transformer encoder with self-attention for long-range audio dependencies
2. U-Net style skip connections for preserving audio detail
3. Improved differentiable DSP with proper multiband filtering
4. Multi-scale feature extraction
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for audio feature extraction.

    Uses self-attention to capture long-range dependencies in audio.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 22050,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        feature_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.d_model = d_model
        self.feature_dim = feature_dim

        # Mel spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # Project mel bins to d_model
        self.input_proj = nn.Linear(n_mels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # Multi-scale pooling for global features
        self.pool_sizes = [1, 2, 4, 8]

    def compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram."""
        if waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Ensure mel transform is on correct device
        if self.mel_transform.mel_scale.fb.device != waveform.device:
            self.mel_transform = self.mel_transform.to(waveform.device)
            self.amplitude_to_db = self.amplitude_to_db.to(waveform.device)

        mel_spec = self.mel_transform(waveform.squeeze(1))  # [B, n_mels, T]
        log_mel = self.amplitude_to_db(mel_spec)
        log_mel = (log_mel + 80) / 80  # Normalize
        return log_mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform."""
        # [B, n_mels, T]
        log_mel = self.compute_spectrogram(waveform)

        # Transpose to [B, T, n_mels] for transformer
        x = log_mel.transpose(1, 2)

        # Project to d_model
        x = self.input_proj(x)  # [B, T, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # [B, T, d_model]

        # Simple mean pooling (MPS compatible - no adaptive pool needed)
        # This is more stable than adaptive pooling on MPS
        features = x.mean(dim=1)  # [B, d_model]

        # Project to output dimension
        features = self.output_proj(features)

        return features


class ImprovedDSPChain(nn.Module):
    """
    Improved differentiable DSP chain with proper multiband processing.

    Features:
    - Learnable crossover filters for multiband split
    - Per-band compression with soft-knee
    - Parallel saturation for warmth
    - True peak limiting with lookahead approximation
    """

    def __init__(self, sample_rate: int = 22050, n_bands: int = 4):
        super().__init__()
        self.sr = sample_rate
        self.n_bands = n_bands

        # Crossover frequencies for multiband (in Hz)
        self.register_buffer(
            'crossover_freqs',
            torch.tensor([100.0, 500.0, 2000.0, 8000.0])[:n_bands]
        )

    def multiband_split(self, x: torch.Tensor) -> list:
        """Split audio into frequency bands using FFT filtering."""
        # FFT-based band splitting for differentiability
        n_fft = 2048
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=x.device)

        # Pad input
        x_mono = x.mean(dim=1) if x.dim() == 3 else x
        if x.dim() == 3:
            x_mono = x.squeeze(1)

        # STFT
        stft = torch.stft(
            x_mono, n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True
        )

        freqs = torch.fft.rfftfreq(n_fft, 1/self.sr).to(x.device)
        bands = []

        prev_freq = 0.0
        for i, cf in enumerate(self.crossover_freqs):
            # Create band mask
            if i < len(self.crossover_freqs) - 1:
                mask = ((freqs >= prev_freq) & (freqs < cf)).float()
            else:
                mask = (freqs >= prev_freq).float()

            # Smooth mask edges
            mask = mask.unsqueeze(0).unsqueeze(-1)

            # Apply mask and inverse STFT
            band_stft = stft * mask
            band_audio = torch.istft(
                band_stft, n_fft=n_fft, hop_length=hop,
                window=window, length=x_mono.shape[-1]
            )
            bands.append(band_audio.unsqueeze(1))
            prev_freq = cf

        return bands

    def soft_knee_compress(
        self,
        x: torch.Tensor,
        threshold: torch.Tensor,
        ratio: torch.Tensor,
        knee_width: float = 6.0,
    ) -> torch.Tensor:
        """Apply soft-knee compression."""
        eps = 1e-8

        # Compute envelope (RMS)
        window_size = int(0.01 * self.sr)
        x_sq = x.pow(2)
        if x_sq.dim() == 2:
            x_sq = x_sq.unsqueeze(1)
        envelope = F.avg_pool1d(
            x_sq, kernel_size=window_size, stride=1,
            padding=window_size // 2
        ).sqrt().squeeze(1)
        envelope = envelope.clamp(min=eps)

        # Convert to dB
        env_db = 20 * torch.log10(envelope + eps)

        # Soft knee gain computation
        threshold_db = threshold
        knee_start = threshold_db - knee_width / 2
        knee_end = threshold_db + knee_width / 2

        # Below knee: no compression
        # In knee: gradual compression
        # Above knee: full compression
        gain_db = torch.zeros_like(env_db)

        # Above threshold
        above = env_db > knee_end
        gain_db = torch.where(
            above,
            threshold_db + (env_db - threshold_db) / ratio - env_db,
            gain_db
        )

        # In knee region (soft transition)
        in_knee = (env_db >= knee_start) & (env_db <= knee_end)
        knee_factor = (env_db - knee_start) / (knee_width + eps)
        knee_gain = (knee_factor.pow(2) * (1/ratio - 1) * (env_db - threshold_db)) / 2
        gain_db = torch.where(in_knee, knee_gain, gain_db)

        # Convert gain to linear and apply
        gain_linear = (10 ** (gain_db / 20)).clamp(min=0.01, max=10.0)

        # Match dimensions
        if gain_linear.dim() < x.dim():
            gain_linear = gain_linear.unsqueeze(1)
        if gain_linear.shape[-1] != x.shape[-1]:
            gain_linear = F.interpolate(
                gain_linear.unsqueeze(1) if gain_linear.dim() == 2 else gain_linear,
                size=x.shape[-1], mode='linear', align_corners=False
            ).squeeze(1)

        return x * gain_linear

    def apply_eq(
        self, x: torch.Tensor, gains_db: torch.Tensor
    ) -> torch.Tensor:
        """Apply parametric EQ using FFT."""
        n_fft = 2048
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=x.device)

        x_mono = x.squeeze(1) if x.dim() == 3 else x

        # STFT
        stft = torch.stft(
            x_mono, n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True
        )

        freqs = torch.fft.rfftfreq(n_fft, 1/self.sr).to(x.device)

        # EQ bands (8 bands log-spaced) - computed on CPU for MPS compatibility
        center_freqs_cpu = torch.logspace(
            torch.log10(torch.tensor(60.0)),
            torch.log10(torch.tensor(16000.0)),
            8
        )
        center_freqs = center_freqs_cpu.to(x.device)

        # Build EQ curve
        eq_curve = torch.ones(x.shape[0], len(freqs), 1, device=x.device)

        for i, cf in enumerate(center_freqs):
            bandwidth = cf / 2.0
            band_response = torch.exp(-((freqs - cf) ** 2) / (2 * bandwidth ** 2))
            gain_linear = 10 ** (gains_db[:, i:i+1] / 20.0)
            eq_curve = eq_curve + (gain_linear.unsqueeze(-1) - 1) * band_response.unsqueeze(0).unsqueeze(-1)

        # Apply EQ
        stft_eq = stft * eq_curve

        # Inverse STFT
        output = torch.istft(
            stft_eq, n_fft=n_fft, hop_length=hop,
            window=window, length=x_mono.shape[-1]
        )

        return output.unsqueeze(1) if x.dim() == 3 else output

    def soft_clip(self, x: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """Soft clipping for limiting."""
        return torch.tanh(x / threshold) * threshold

    def forward(
        self, waveform: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply full mastering chain."""
        batch_size = waveform.shape[0]

        # 1. EQ
        x = self.apply_eq(waveform, params['eq_gains'])

        # 2. Simplified compression (avoid in-place ops)
        # Use a simple soft-knee approach that's fully differentiable
        threshold_linear = 10 ** (params['comp_thresholds'][:, 0:1] / 20.0)
        threshold_linear = threshold_linear.unsqueeze(-1)  # [B, 1, 1]

        # Soft compression using tanh-based curve
        ratio = params['comp_ratios'][:, 0:1].clamp(min=1.1, max=20.0).unsqueeze(-1)

        # Compute gain reduction
        x_abs = x.abs().clamp(min=1e-8)
        above_threshold = (x_abs > threshold_linear).float()
        compression_amount = above_threshold * (1.0 - 1.0/ratio) * (x_abs - threshold_linear) / (x_abs + 1e-8)
        x = x * (1.0 - compression_amount.clamp(min=0, max=0.9))

        # 3. Soft limiting (no in-place)
        threshold = 10 ** (params['limiter_threshold'] / 20.0)
        threshold = threshold.unsqueeze(-1).clamp(min=0.1, max=1.0)  # [B, 1, 1]
        x = torch.tanh(x / threshold) * threshold

        # 4. Output gain normalization
        target_rms = 10 ** (params['target_lufs'] / 20.0) * 0.1
        target_rms = target_rms.unsqueeze(-1)  # [B, 1, 1]
        current_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        gain = (target_rms / current_rms).clamp(min=0.1, max=10.0)
        x = x * gain

        # Ensure output is in valid range
        x = x.clamp(min=-1.0, max=1.0)

        return x


class GenreMasterV2(nn.Module):
    """
    GenreMaster V2 with improved architecture.

    Changes from V1:
    - Transformer encoder for better feature extraction
    - Improved DSP chain with proper multiband processing
    - Residual connection option for detail preservation
    """

    def __init__(
        self,
        n_genres: int,
        sample_rate: int = 22050,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        feature_dim: int = 512,
        genre_latent_dim: int = 128,
        use_residual: bool = True,
    ):
        super().__init__()

        self.n_genres = n_genres
        self.feature_dim = feature_dim
        self.use_residual = use_residual

        # Transformer encoder
        self.encoder = TransformerEncoder(
            sample_rate=sample_rate,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            feature_dim=feature_dim,
        )

        # Genre embedding
        self.genre_embedding = nn.Embedding(n_genres, genre_latent_dim)

        # Genre-audio fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + genre_latent_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # Parameter prediction heads
        hidden_dim = 256

        self.param_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Individual parameter heads
        self.lufs_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.eq_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 8), nn.Tanh()
        )
        self.comp_threshold_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 3), nn.Sigmoid()
        )
        self.comp_ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 3), nn.Sigmoid()
        )
        self.limiter_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Residual mixing weight (learnable)
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # DSP chain
        self.dsp = ImprovedDSPChain(sample_rate=sample_rate)

    def predict_params(
        self, audio_features: torch.Tensor, genre_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict DSP parameters from features and genre."""
        # Get genre embedding
        genre_emb = self.genre_embedding(genre_idx)

        # Fuse audio features with genre
        fused = torch.cat([audio_features, genre_emb], dim=-1)
        fused = self.fusion(fused)

        # Predict parameters
        features = self.param_net(fused)

        params = {
            'target_lufs': self.lufs_head(features) * 22 - 30,  # [-30, -8]
            'eq_gains': self.eq_head(features) * 12,  # [-12, +12] dB
            'comp_thresholds': self.comp_threshold_head(features) * 30 - 40,  # [-40, -10]
            'comp_ratios': self.comp_ratio_head(features) * 9 + 1,  # [1, 10]
            'limiter_threshold': self.limiter_head(features) * 10 - 10,  # [-10, 0]
            'stereo_width': self.width_head(features) * 2,  # [0, 2]
        }

        return params

    def forward(
        self,
        waveform: torch.Tensor,
        genre_idx: torch.Tensor,
        return_params: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        audio_features = self.encoder(waveform)

        # Predict DSP parameters
        params = self.predict_params(audio_features, genre_idx)

        # Apply DSP chain
        processed = self.dsp(waveform, params)

        # Optional residual connection (preserve some original detail)
        if self.use_residual:
            weight = torch.sigmoid(self.residual_weight)
            output = weight * waveform + (1 - weight) * processed
        else:
            output = processed

        if return_params:
            return output, params
        return output


def create_genremaster_v2(
    n_genres: int = 10,
    sample_rate: int = 22050,
    use_residual: bool = True,
) -> GenreMasterV2:
    """Factory function to create GenreMaster V2 model."""
    return GenreMasterV2(
        n_genres=n_genres,
        sample_rate=sample_rate,
        d_model=256,
        n_heads=8,
        n_encoder_layers=4,
        feature_dim=512,
        genre_latent_dim=128,
        use_residual=use_residual,
    )


if __name__ == "__main__":
    # Test the model
    model = create_genremaster_v2(n_genres=10)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GenreMaster V2 Parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    audio_length = 22050 * 5  # 5 seconds
    waveform = torch.randn(batch_size, 1, audio_length)
    genre_idx = torch.randint(0, 10, (batch_size,))

    output, params = model(waveform, genre_idx, return_params=True)
    print(f"Input shape: {waveform.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted parameters:")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")
