"""Genre embedding network with contrastive learning.

Learns genre-conditioned latent representations for mastering control.
Uses contrastive learning to ensure genre embeddings capture semantic differences.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreEmbeddingNetwork(nn.Module):
    """
    Genre embedding network with audio-conditioned refinement.

    Architecture:
    1. Genre label (class index) → learnable embedding
    2. Audio features → projection
    3. Concatenate and refine via MLP
    4. Output: genre-conditioned latent z_g ∈ R^128
    """

    def __init__(
        self,
        n_genres: int,
        embedding_dim: int = 256,
        audio_feature_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize genre embedding network.

        Args:
            n_genres: Number of genre classes
            embedding_dim: Genre embedding dimension
            audio_feature_dim: Dimension of audio features from encoder
            output_dim: Output latent dimension (z_g)
            dropout: Dropout rate
        """
        super().__init__()

        self.n_genres = n_genres
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Learnable genre embeddings
        self.genre_embedding = nn.Embedding(n_genres, embedding_dim)

        # Audio feature projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Refinement MLP: [genre_emb + audio_proj] → z_g
        self.refinement = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        genre_idx: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: genre index + audio features → latent z_g.

        Args:
            genre_idx: Genre class indices of shape [batch]
            audio_features: Audio features of shape [batch, audio_feature_dim]
                           If None, uses genre embedding only

        Returns:
            Genre-conditioned latent z_g of shape [batch, output_dim]
        """
        # Get genre embeddings
        genre_emb = self.genre_embedding(genre_idx)  # [batch, embedding_dim]

        if audio_features is None:
            # Genre-only mode (no audio conditioning)
            # Pass through refinement with repeated genre embedding
            combined = torch.cat([genre_emb, genre_emb], dim=1)
        else:
            # Project audio features
            audio_proj = self.audio_proj(audio_features)  # [batch, embedding_dim]

            # Concatenate genre and audio
            combined = torch.cat([genre_emb, audio_proj], dim=1)

        # Refine to output latent
        z_g = self.refinement(combined)  # [batch, output_dim]

        return z_g

    def get_genre_embedding(self, genre_idx: torch.Tensor) -> torch.Tensor:
        """
        Get raw genre embedding without audio conditioning.

        Args:
            genre_idx: Genre class indices of shape [batch]

        Returns:
            Genre embeddings of shape [batch, embedding_dim]
        """
        return self.genre_embedding(genre_idx)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for genre embedding pre-training.

    Uses InfoNCE (NT-Xent) loss to ensure:
    - Tracks of the same genre are pulled together in embedding space
    - Tracks of different genres are pushed apart
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for softmax scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Genre embeddings of shape [batch, dim]
            labels: Genre labels of shape [batch]

        Returns:
            Contrastive loss scalar
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same genre)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float().to(device)

        # Remove self-similarity on diagonal
        positive_mask.fill_diagonal_(0)

        # Create mask for negative pairs (different genre)
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)

        # For each anchor, compute loss
        losses = []
        for i in range(batch_size):
            # Get positive pairs for this anchor
            positives = similarity_matrix[i][positive_mask[i] == 1]

            if len(positives) == 0:
                continue  # Skip if no positive pairs in batch

            # Get negative pairs
            negatives = similarity_matrix[i][negative_mask[i] == 1]

            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            for pos in positives:
                logits = torch.cat([pos.unsqueeze(0), negatives])
                loss = -torch.log_softmax(logits, dim=0)[0]
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


class GenreClassifier(nn.Module):
    """
    Simple genre classifier head for supervised pre-training.

    Can be used alongside contrastive loss for better genre separation.
    """

    def __init__(self, input_dim: int, n_genres: int):
        """
        Initialize genre classifier.

        Args:
            input_dim: Input feature dimension
            n_genres: Number of genre classes
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_genres),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → genre logits.

        Args:
            features: Input features of shape [batch, input_dim]

        Returns:
            Genre logits of shape [batch, n_genres]
        """
        return self.classifier(features)


def create_genre_embedding_network(
    n_genres: int,
    audio_feature_dim: int = 512,
    output_dim: int = 128,
) -> GenreEmbeddingNetwork:
    """
    Factory function to create genre embedding network.

    Args:
        n_genres: Number of genre classes
        audio_feature_dim: Dimension of audio features
        output_dim: Output latent dimension

    Returns:
        GenreEmbeddingNetwork instance
    """
    return GenreEmbeddingNetwork(
        n_genres=n_genres,
        embedding_dim=256,
        audio_feature_dim=audio_feature_dim,
        output_dim=output_dim,
        dropout=0.1,
    )
