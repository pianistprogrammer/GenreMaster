"""Feature-wise Linear Modulation (FiLM) conditioning layer.

FiLM enables genre-conditioned control by modulating features with
learned scale (γ) and shift (β) parameters derived from genre embeddings.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", 2018
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies affine transformation: FiLM(x, z) = γ(z) ⊙ x + β(z)

    where:
    - x: input features
    - z: conditioning vector (genre embedding)
    - γ: scale parameter (learned from z)
    - β: shift parameter (learned from z)
    - ⊙: element-wise multiplication
    """

    def __init__(self, feature_dim: int, conditioning_dim: int):
        """
        Initialize FiLM layer.

        Args:
            feature_dim: Dimension of features to be modulated
            conditioning_dim: Dimension of conditioning vector (z_g)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim

        # Predict scale (gamma) and shift (beta) from conditioning vector
        self.film_generator = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(inplace=True),
            nn.Linear(conditioning_dim, feature_dim * 2),  # *2 for gamma and beta
        )

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to features.

        Args:
            features: Input features of shape [batch, feature_dim] or [batch, feature_dim, ...]
            conditioning: Conditioning vector of shape [batch, conditioning_dim]

        Returns:
            Modulated features of same shape as input
        """
        # Generate gamma and beta
        film_params = self.film_generator(conditioning)  # [batch, feature_dim * 2]

        # Split into scale and shift
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # Each [batch, feature_dim]

        # Reshape for broadcasting if features have spatial dimensions
        original_shape = features.shape
        if len(original_shape) > 2:
            # Flatten spatial dimensions for easier handling
            batch_size = features.shape[0]
            features_flat = features.view(batch_size, self.feature_dim, -1)

            # Expand gamma and beta for broadcasting
            gamma = gamma.unsqueeze(-1)  # [batch, feature_dim, 1]
            beta = beta.unsqueeze(-1)  # [batch, feature_dim, 1]

            # Apply FiLM
            modulated = gamma * features_flat + beta

            # Reshape back to original
            modulated = modulated.view(original_shape)
        else:
            # Simple case: [batch, feature_dim]
            modulated = gamma * features + beta

        return modulated


class MultiLayerFiLM(nn.Module):
    """
    Apply FiLM conditioning to multiple feature layers.

    Useful for conditioning multiple processing stages with the same
    genre embedding (e.g., each DSP module in the mastering chain).
    """

    def __init__(
        self,
        feature_dims: list[int],
        conditioning_dim: int,
        shared_generator: bool = False,
    ):
        """
        Initialize multi-layer FiLM.

        Args:
            feature_dims: List of feature dimensions for each layer
            conditioning_dim: Dimension of conditioning vector
            shared_generator: Share FiLM generator across layers (parameter efficient)
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.conditioning_dim = conditioning_dim
        self.shared_generator = shared_generator

        if shared_generator:
            # Single shared generator with different heads per layer
            self.shared_encoder = nn.Sequential(
                nn.Linear(conditioning_dim, conditioning_dim),
                nn.ReLU(inplace=True),
            )
            self.layer_heads = nn.ModuleList([
                nn.Linear(conditioning_dim, dim * 2) for dim in feature_dims
            ])
        else:
            # Independent FiLM generator for each layer
            self.film_layers = nn.ModuleList([
                FiLMLayer(dim, conditioning_dim) for dim in feature_dims
            ])

    def forward(
        self,
        features_list: list[torch.Tensor],
        conditioning: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Apply FiLM to multiple feature layers.

        Args:
            features_list: List of feature tensors
            conditioning: Conditioning vector [batch, conditioning_dim]

        Returns:
            List of modulated feature tensors
        """
        if self.shared_generator:
            # Shared encoding
            encoded = self.shared_encoder(conditioning)

            modulated_features = []
            for features, head in zip(features_list, self.layer_heads):
                # Generate layer-specific gamma and beta
                film_params = head(encoded)
                gamma, beta = torch.chunk(film_params, 2, dim=1)

                # Apply FiLM
                if len(features.shape) > 2:
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)

                modulated = gamma * features + beta
                modulated_features.append(modulated)

            return modulated_features
        else:
            # Independent FiLM for each layer
            return [
                film_layer(features, conditioning)
                for features, film_layer in zip(features_list, self.film_layers)
            ]


class AdaptiveInstanceNormFiLM(nn.Module):
    """
    FiLM with Adaptive Instance Normalization.

    Combines instance normalization with FiLM for stronger conditioning effect.
    Useful when input features have highly variable scales.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int, eps: float = 1e-5):
        """
        Initialize AdaIN-FiLM layer.

        Args:
            feature_dim: Dimension of features to be modulated
            conditioning_dim: Dimension of conditioning vector
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.eps = eps

        # Generate scale and shift
        self.film_generator = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(inplace=True),
            nn.Linear(conditioning_dim, feature_dim * 2),
        )

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN-FiLM modulation.

        Args:
            features: Input features [batch, feature_dim] or [batch, feature_dim, ...]
            conditioning: Conditioning vector [batch, conditioning_dim]

        Returns:
            Modulated features
        """
        # Instance normalization
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True) + self.eps
        normalized = (features - mean) / std

        # Generate gamma and beta
        film_params = self.film_generator(conditioning)
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        # Reshape for broadcasting if needed
        if len(features.shape) > 2:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        # Apply FiLM to normalized features
        modulated = gamma * normalized + beta

        return modulated


def create_film_layer(
    feature_dim: int,
    conditioning_dim: int,
    use_adain: bool = False,
) -> nn.Module:
    """
    Factory function to create FiLM layer.

    Args:
        feature_dim: Dimension of features to modulate
        conditioning_dim: Dimension of conditioning vector
        use_adain: Use adaptive instance normalization variant

    Returns:
        FiLM layer module
    """
    if use_adain:
        return AdaptiveInstanceNormFiLM(feature_dim, conditioning_dim)
    else:
        return FiLMLayer(feature_dim, conditioning_dim)
