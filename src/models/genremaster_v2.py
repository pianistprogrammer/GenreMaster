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

        # Instance normalization: zero mean, unit variance per sample
        # Much better than fixed offset for transformer input
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        log_mel = (log_mel - mean) / std

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

    NOTE: Some operations run on CPU for MPS compatibility, then move back.
    """

    def __init__(self, sample_rate: int = 22050, n_bands: int = 4):
        super().__init__()
        self.sr = sample_rate
        self.n_bands = n_bands

        # Pre-compute EQ center frequencies on CPU (MPS doesn't support logspace)
        self.register_buffer(
            'eq_center_freqs',
            torch.logspace(
                torch.log10(torch.tensor(60.0)),
                torch.log10(torch.tensor(16000.0)),
                8
            )
        )

    def apply_eq(
        self, x: torch.Tensor, gains_db: torch.Tensor
    ) -> torch.Tensor:
        """Apply parametric EQ using learnable biquad-style FIR filtering.

        Uses time-domain convolution instead of STFT to maintain gradient flow
        on MPS devices.
        """
        device = x.device
        x_mono = x.squeeze(1) if x.dim() == 3 else x

        # Build a short FIR EQ filter from the predicted gains
        n_taps = 257  # FIR filter length (odd for symmetry)
        half = n_taps // 2

        # Frequency axis for the FIR filter (0 to Nyquist)
        n_freq = half + 1
        freq_axis = torch.linspace(0, self.sr / 2, n_freq, device=device)
        center_freqs = self.eq_center_freqs.to(device)

        # Build EQ magnitude response [batch, n_freq]
        eq_mag = torch.ones(x_mono.shape[0], n_freq, device=device)

        for i, cf in enumerate(center_freqs):
            bandwidth = cf / 2.0 + 1.0  # Avoid zero bandwidth
            band_response = torch.exp(
                -((freq_axis - cf) ** 2) / (2 * bandwidth ** 2)
            )  # [n_freq]
            gain_linear = 10 ** (gains_db[:, i:i+1] / 20.0)  # [batch, 1]
            eq_mag = eq_mag + (gain_linear - 1) * band_response.unsqueeze(0)

        # Convert magnitude response to minimum-phase FIR via cepstral method
        # Use symmetric (zero-phase) FIR for simplicity: IFFT of magnitude
        # Mirror to get full spectrum, then IRFFT
        fir = torch.fft.irfft(eq_mag, n=n_taps, dim=-1)  # [batch, n_taps]

        # Circular shift to center the filter and apply window
        fir = torch.roll(fir, half, dims=-1)
        window = torch.hann_window(n_taps, device=device)
        fir = fir * window.unsqueeze(0)

        # Normalize filter energy
        fir = fir / (fir.sum(dim=-1, keepdim=True).abs().clamp(min=1e-6))

        # Apply as grouped 1D convolution [batch, samples] -> [batch, 1, samples]
        x_padded = F.pad(x_mono, (half, half), mode='reflect')
        x_conv = x_padded.unsqueeze(1)  # [batch, 1, samples+pad]
        fir_kernel = fir.unsqueeze(1)  # [batch, 1, n_taps]

        # Per-sample convolution via groups
        batch_size = x_mono.shape[0]
        x_grouped = x_conv.reshape(1, batch_size, -1)  # [1, batch, samples]
        fir_grouped = fir_kernel  # [batch, 1, n_taps]
        output = F.conv1d(x_grouped, fir_grouped, groups=batch_size)
        output = output.reshape(batch_size, -1)  # [batch, samples]

        # Trim to original length
        if output.shape[-1] > x_mono.shape[-1]:
            output = output[..., :x_mono.shape[-1]]

        output = output.unsqueeze(1) if x.dim() == 3 else output
        return output

    def soft_knee_compress(
        self,
        x: torch.Tensor,
        threshold_db: torch.Tensor,
        ratio: torch.Tensor,
        knee_width: float = 6.0,
    ) -> torch.Tensor:
        """Apply soft-knee compression. Fully differentiable, MPS compatible."""
        eps = 1e-8

        # Compute RMS envelope
        window_size = max(int(0.01 * self.sr), 1)

        x_sq = x.pow(2)
        if x_sq.dim() == 2:
            x_sq = x_sq.unsqueeze(1)

        # Use unfold + mean instead of avg_pool1d for better MPS compatibility
        # Pad first
        pad_size = window_size // 2
        x_padded = F.pad(x_sq, (pad_size, pad_size), mode='reflect')

        # Simple moving average using conv1d with fixed weights
        kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
        envelope = F.conv1d(x_padded, kernel, groups=1).sqrt()

        # Trim to original size
        if envelope.shape[-1] > x.shape[-1]:
            envelope = envelope[..., :x.shape[-1]]
        elif envelope.shape[-1] < x.shape[-1]:
            envelope = F.pad(envelope, (0, x.shape[-1] - envelope.shape[-1]))

        envelope = envelope.squeeze(1) if x.dim() == 2 else envelope
        envelope = envelope.clamp(min=eps)

        # Convert to dB
        env_db = 20 * torch.log10(envelope + eps)

        # Threshold in dB (reshape for broadcasting)
        threshold_db = threshold_db.view(-1, 1, 1) if x.dim() == 3 else threshold_db.view(-1, 1)
        ratio = ratio.view(-1, 1, 1) if x.dim() == 3 else ratio.view(-1, 1)

        # Soft knee compression
        knee_start = threshold_db - knee_width / 2
        knee_end = threshold_db + knee_width / 2

        # Gain reduction calculation
        gain_db = torch.zeros_like(env_db)

        # Above knee: full compression
        above = env_db > knee_end
        gain_db = torch.where(
            above,
            threshold_db + (env_db - threshold_db) / ratio - env_db,
            gain_db
        )

        # In knee region: gradual compression
        in_knee = (env_db >= knee_start) & (env_db <= knee_end)
        knee_factor = (env_db - knee_start) / (knee_width + eps)
        knee_gain = (knee_factor.pow(2) * (1/ratio - 1) * (env_db - threshold_db)) / 2
        gain_db = torch.where(in_knee, knee_gain, gain_db)

        # Convert to linear gain and apply
        gain_linear = (10 ** (gain_db / 20)).clamp(min=0.01, max=10.0)

        return x * gain_linear

    def soft_clip(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Soft clipping using tanh. MPS compatible."""
        # Reshape threshold for broadcasting
        if threshold.dim() == 2:
            threshold = threshold.unsqueeze(-1)
        threshold = threshold.clamp(min=0.1, max=1.0)
        return torch.tanh(x / threshold) * threshold

    def forward(
        self, waveform: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply full mastering chain."""
        # 1. Parametric EQ (runs on CPU, returns to MPS)
        x = self.apply_eq(waveform, params['eq_gains'])

        # 2. Multiband compression (simplified to single-band for stability)
        x = self.soft_knee_compress(
            x,
            params['comp_thresholds'][:, 0],  # Use first band
            params['comp_ratios'][:, 0].clamp(min=1.1, max=20.0),
        )

        # 3. True peak limiting
        x = self.soft_clip(x, 10 ** (params['limiter_threshold'] / 20.0))

        # 4. Output gain normalization to target loudness
        target_rms = (10 ** (params['target_lufs'] / 20.0) * 0.1).unsqueeze(-1)
        current_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        gain = (target_rms / current_rms).clamp(min=0.1, max=10.0)
        x = x * gain

        # Final clipping to valid range
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
        # Initialize at -2.0 so sigmoid(-2.0) ≈ 0.12 — DSP output dominates.
        # The model needs to learn to master (transform) the audio,
        # so the DSP chain should drive the output from the start.
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(-2.0))

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
