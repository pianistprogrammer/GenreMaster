"""GenreMaster: Complete genre-conditioned automatic mastering model.

Assembles all components into an end-to-end trainable model:
- Spectral Encoder (audio → features)
- Genre Embedding Network (genre + audio → z_g)
- FiLM-conditioned Parameter Prediction
- Differentiable Mastering Chain (DSP modules)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder import create_encoder
from .genre_embedding import GenreEmbeddingNetwork
from .film import FiLMLayer
from .dsp_modules import MasteringChain


class ParameterPredictor(nn.Module):
    """
    Predicts DSP parameters from audio features and genre conditioning.

    Uses FiLM to modulate audio features with genre embedding,
    then predicts parameters for each DSP module.
    """

    def __init__(
        self,
        audio_feature_dim: int = 512,
        genre_latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        """
        Initialize parameter predictor.

        Args:
            audio_feature_dim: Dimension of audio features from encoder
            genre_latent_dim: Dimension of genre latent z_g
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # FiLM conditioning for audio features
        self.film = FiLMLayer(
            feature_dim=audio_feature_dim,
            conditioning_dim=genre_latent_dim,
        )

        # Shared feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Parameter prediction heads for each DSP module
        # 1. Loudness normalization: target LUFS
        self.lufs_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Map to [0, 1]
        )

        # 2. Parametric EQ: 8-band gains
        self.eq_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
            nn.Tanh(),  # Map to [-1, 1]
        )

        # 3. Multiband compressor: 3-band thresholds and ratios
        self.comp_threshold_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        self.comp_ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        # 4. True peak limiter: threshold
        self.limiter_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 5. Stereo width: width factor
        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        genre_latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict DSP parameters from features and genre.

        Args:
            audio_features: Audio features [batch, audio_feature_dim]
            genre_latent: Genre latent z_g [batch, genre_latent_dim]

        Returns:
            Dictionary of DSP parameters
        """
        # Apply FiLM conditioning
        conditioned_features = self.film(audio_features, genre_latent)

        # Process features
        features = self.feature_net(conditioned_features)

        # Predict parameters with proper ranges
        # LUFS: [-30, -8] for various genres
        target_lufs = self.lufs_head(features) * 22 - 30  # [0,1] → [-30, -8]

        # EQ gains: [-12, +12] dB per band
        eq_gains = self.eq_head(features) * 12  # [-1,1] → [-12, +12]

        # Compression thresholds: [-40, -10] dB
        comp_thresholds = self.comp_threshold_head(features) * 30 - 40  # [0,1] → [-40, -10]

        # Compression ratios: [1, 10]
        comp_ratios = self.comp_ratio_head(features) * 9 + 1  # [0,1] → [1, 10]

        # Limiter threshold: [-10, 0] dB
        limiter_threshold = self.limiter_head(features) * 10 - 10  # [0,1] → [-10, 0]

        # Stereo width: [0, 2] (0=mono, 1=unchanged, 2=wide)
        stereo_width = self.width_head(features) * 2  # [0,1] → [0, 2]

        return {
            'target_lufs': target_lufs,
            'eq_gains': eq_gains,
            'comp_thresholds': comp_thresholds,
            'comp_ratios': comp_ratios,
            'limiter_threshold': limiter_threshold,
            'stereo_width': stereo_width,
        }


class GenreMaster(nn.Module):
    """
    Complete GenreMaster model for genre-conditioned automatic mastering.

    Architecture:
        Input Audio → Spectral Encoder → Audio Features
                                              ↓
        Genre Label → Genre Embedding ──→ z_g (genre latent)
                                              ↓
                                    [FiLM Conditioning]
                                              ↓
                                    Parameter Predictor
                                              ↓
                                      DSP Parameters
                                              ↓
                                    Mastering Chain
                                              ↓
                                    Mastered Output
    """

    def __init__(
        self,
        n_genres: int,
        encoder_type: str = "lightweight",
        audio_feature_dim: int = 512,
        genre_latent_dim: int = 128,
        sample_rate: int = 44100,
    ):
        """
        Initialize GenreMaster model.

        Args:
            n_genres: Number of genre classes
            encoder_type: Type of spectral encoder ('resnet' or 'lightweight')
            audio_feature_dim: Dimension of audio features
            genre_latent_dim: Dimension of genre latent z_g
            sample_rate: Audio sample rate
        """
        super().__init__()

        self.n_genres = n_genres
        self.audio_feature_dim = audio_feature_dim
        self.genre_latent_dim = genre_latent_dim
        self.sr = sample_rate

        # 1. Spectral Encoder: audio → features
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            feature_dim=audio_feature_dim,
            sample_rate=sample_rate,
        )

        # 2. Genre Embedding Network: genre + audio → z_g
        self.genre_embedding = GenreEmbeddingNetwork(
            n_genres=n_genres,
            embedding_dim=256,
            audio_feature_dim=audio_feature_dim,
            output_dim=genre_latent_dim,
        )

        # 3. Parameter Predictor: features + z_g → DSP params
        self.param_predictor = ParameterPredictor(
            audio_feature_dim=audio_feature_dim,
            genre_latent_dim=genre_latent_dim,
            hidden_dim=256,
        )

        # 4. Differentiable Mastering Chain: audio + params → mastered audio
        self.mastering_chain = MasteringChain(sample_rate=sample_rate)

    def forward(
        self,
        waveform: torch.Tensor,
        genre_idx: torch.Tensor,
        return_params: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: waveform + genre → mastered waveform.

        Args:
            waveform: Input audio [batch, channels, samples]
            genre_idx: Genre class indices [batch]
            return_params: If True, also return predicted DSP parameters

        Returns:
            Mastered audio [batch, channels, samples]
            (optionally) Dictionary of predicted DSP parameters
        """
        # 1. Extract audio features
        audio_features = self.encoder(waveform)  # [batch, audio_feature_dim]

        # 2. Get genre-conditioned latent
        genre_latent = self.genre_embedding(
            genre_idx, audio_features
        )  # [batch, genre_latent_dim]

        # 3. Predict DSP parameters
        dsp_params = self.param_predictor(
            audio_features, genre_latent
        )  # Dict of parameters

        # 4. Apply mastering chain
        mastered = self.mastering_chain(waveform, dsp_params)

        if return_params:
            return mastered, dsp_params
        return mastered

    def forward_with_genre_only(
        self,
        waveform: torch.Tensor,
        genre_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using genre embedding only (no audio conditioning).

        Useful for testing genre-specific processing without audio features.

        Args:
            waveform: Input audio [batch, channels, samples]
            genre_idx: Genre class indices [batch]

        Returns:
            Mastered audio [batch, channels, samples]
        """
        # Extract audio features (still needed for encoder)
        audio_features = self.encoder(waveform)

        # Get genre latent WITHOUT audio conditioning
        genre_latent = self.genre_embedding(genre_idx, audio_features=None)

        # Predict DSP parameters
        dsp_params = self.param_predictor(audio_features, genre_latent)

        # Apply mastering chain
        mastered = self.mastering_chain(waveform, dsp_params)

        return mastered

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter counts for each component.

        Returns:
            Dictionary of parameter counts
        """
        counts = {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'genre_embedding': sum(p.numel() for p in self.genre_embedding.parameters()),
            'param_predictor': sum(p.numel() for p in self.param_predictor.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }
        return counts


class GenreMasterUnconditioned(nn.Module):
    """
    Baseline GenreMaster WITHOUT genre conditioning.

    Used for ablation studies to measure the contribution of genre conditioning.
    """

    def __init__(
        self,
        encoder_type: str = "lightweight",
        audio_feature_dim: int = 512,
        sample_rate: int = 44100,
    ):
        """
        Initialize unconditioned baseline.

        Args:
            encoder_type: Type of spectral encoder
            audio_feature_dim: Dimension of audio features
            sample_rate: Audio sample rate
        """
        super().__init__()

        self.encoder = create_encoder(
            encoder_type=encoder_type,
            feature_dim=audio_feature_dim,
            sample_rate=sample_rate,
        )

        # Parameter predictor WITHOUT FiLM conditioning
        self.param_predictor = nn.Sequential(
            nn.Linear(audio_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Parameter heads (same as conditioned model)
        self.lufs_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.eq_head = nn.Sequential(nn.Linear(128, 8), nn.Tanh())
        self.comp_threshold_head = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())
        self.comp_ratio_head = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())
        self.limiter_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.width_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        self.mastering_chain = MasteringChain(sample_rate=sample_rate)

    def forward(self, waveform: torch.Tensor, genre_idx: torch.Tensor = None) -> torch.Tensor:
        """Forward pass (genre_idx ignored for compatibility)."""
        audio_features = self.encoder(waveform)
        features = self.param_predictor(audio_features)

        # Predict parameters
        target_lufs = self.lufs_head(features) * 22 - 30
        eq_gains = self.eq_head(features) * 12
        comp_thresholds = self.comp_threshold_head(features) * 30 - 40
        comp_ratios = self.comp_ratio_head(features) * 9 + 1
        limiter_threshold = self.limiter_head(features) * 10 - 10
        stereo_width = self.width_head(features) * 2

        dsp_params = {
            'target_lufs': target_lufs,
            'eq_gains': eq_gains,
            'comp_thresholds': comp_thresholds,
            'comp_ratios': comp_ratios,
            'limiter_threshold': limiter_threshold,
            'stereo_width': stereo_width,
        }

        mastered = self.mastering_chain(waveform, dsp_params)
        return mastered


def create_genremaster_model(
    n_genres: int,
    encoder_type: str = "lightweight",
    conditioned: bool = True,
    sample_rate: int = 44100,
) -> nn.Module:
    """
    Factory function to create GenreMaster model.

    Args:
        n_genres: Number of genre classes
        encoder_type: 'resnet' or 'lightweight'
        conditioned: If True, use genre conditioning; if False, use baseline
        sample_rate: Audio sample rate

    Returns:
        GenreMaster or GenreMasterUnconditioned model
    """
    if conditioned:
        return GenreMaster(
            n_genres=n_genres,
            encoder_type=encoder_type,
            sample_rate=sample_rate,
        )
    else:
        return GenreMasterUnconditioned(
            encoder_type=encoder_type,
            sample_rate=sample_rate,
        )
