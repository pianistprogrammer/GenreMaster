"""Pre-trained genre classifier wrapper for GenreMaster.

Integrates SOTA pre-trained genre classifiers (Essentia MusicNN, etc.)
into the GenreMaster pipeline for automatic genre detection.
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import numpy as np


class PretrainedGenreClassifier(nn.Module):
    """
    Wrapper for pre-trained genre classification models.

    Supports:
    - Essentia MusicNN (recommended)
    - Custom PyTorch models
    - Hugging Face models
    """

    def __init__(
        self,
        model_type: str = "essentia",
        num_genres: int = 16,
        fma_genres: Optional[List[int]] = None,
    ):
        """
        Initialize pre-trained genre classifier.

        Args:
            model_type: Type of pre-trained model ("essentia", "pytorch", "huggingface")
            num_genres: Number of genres in your dataset
            fma_genres: List of FMA genre IDs to map to
        """
        super().__init__()

        self.model_type = model_type
        self.num_genres = num_genres
        self.fma_genres = fma_genres

        if model_type == "essentia":
            self._init_essentia()
        elif model_type == "pytorch":
            self._init_pytorch()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _init_essentia(self):
        """Initialize Essentia MusicNN model."""
        try:
            import essentia.standard as es

            # Load pre-trained MusicNN genre classifier
            # Download from: https://essentia.upf.edu/models/
            model_path = "models/genre-musicnn-msd.pb"
            self.essentia_model = es.TensorflowPredictMusiCNN(
                graphFilename=model_path,
                output="model/dense/BiasAdd"
            )

            # Essentia models output different genre taxonomy than FMA
            # Need to map Essentia genres → FMA genres
            self.genre_mapping = self._create_genre_mapping()

        except ImportError:
            raise ImportError(
                "Essentia not installed. Install with: pip install essentia-tensorflow"
            )

    def _init_pytorch(self):
        """Initialize PyTorch pre-trained model."""
        # Placeholder for custom PyTorch models
        # Could load a model from torch.hub or local checkpoint
        pass

    def _create_genre_mapping(self) -> Dict[int, int]:
        """
        Map Essentia genre predictions to FMA genre IDs.

        Essentia MusicNN outputs 50 genre classes.
        We need to map these to our FMA 16 genres.

        Returns:
            Dictionary mapping Essentia genre idx → FMA genre idx
        """
        # This is a simplified mapping - you'd need to create
        # a proper mapping based on genre taxonomies
        essentia_to_fma = {
            0: 13,  # Rock → Rock
            1: 4,   # Electronic → Electronic
            2: 7,   # Hip-Hop → Hip-Hop
            3: 10,  # Jazz → Jazz
            # ... complete mapping for all genres
        }
        return essentia_to_fma

    def forward(
        self,
        waveform: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Predict genre from audio waveform.

        Args:
            waveform: Audio tensor [batch, channels, samples]
            return_probs: If True, return probabilities instead of class indices

        Returns:
            Genre indices [batch] or probabilities [batch, num_genres]
        """
        if self.model_type == "essentia":
            return self._predict_essentia(waveform, return_probs)
        else:
            raise NotImplementedError(f"Prediction not implemented for {self.model_type}")

    def _predict_essentia(
        self,
        waveform: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Predict using Essentia model."""
        import essentia.standard as es

        batch_size = waveform.shape[0]
        predictions = []

        for i in range(batch_size):
            # Convert to mono and numpy
            audio_np = waveform[i].mean(dim=0).cpu().numpy()

            # Essentia expects 16kHz mono
            # Resample if needed
            if waveform.shape[-1] != 16000 * 30:  # 30 seconds at 16kHz
                resampler = es.Resample(
                    inputSampleRate=44100,
                    outputSampleRate=16000
                )
                audio_np = resampler(audio_np)

            # Get predictions
            probs = self.essentia_model(audio_np)

            # Map to FMA genres
            fma_probs = self._map_to_fma_genres(probs)
            predictions.append(fma_probs)

        predictions = torch.tensor(np.array(predictions), device=waveform.device)

        if return_probs:
            return predictions
        else:
            return predictions.argmax(dim=1)

    def _map_to_fma_genres(self, essentia_probs: np.ndarray) -> np.ndarray:
        """Map Essentia predictions to FMA genre space."""
        # Create FMA probability distribution
        fma_probs = np.zeros(self.num_genres)

        # Aggregate Essentia predictions into FMA categories
        # This is simplified - you'd want a proper mapping
        for essentia_idx, prob in enumerate(essentia_probs):
            if essentia_idx in self.genre_mapping:
                fma_idx = self.genre_mapping[essentia_idx]
                fma_probs[fma_idx] += prob

        # Normalize
        fma_probs = fma_probs / (fma_probs.sum() + 1e-8)

        return fma_probs


class GenreMasterWithPretrainedClassifier(nn.Module):
    """
    GenreMaster model with integrated pre-trained genre classifier.

    This version automatically detects genre from audio and uses it
    for conditioning, while still allowing manual genre specification.
    """

    def __init__(
        self,
        genremaster_model: nn.Module,
        classifier_type: str = "essentia",
        num_genres: int = 16,
    ):
        """
        Initialize GenreMaster with pre-trained classifier.

        Args:
            genremaster_model: Existing GenreMaster model
            classifier_type: Type of pre-trained classifier
            num_genres: Number of genres
        """
        super().__init__()

        self.genremaster = genremaster_model
        self.genre_classifier = PretrainedGenreClassifier(
            model_type=classifier_type,
            num_genres=num_genres,
        )

        # Freeze classifier (don't train it)
        for param in self.genre_classifier.parameters():
            param.requires_grad = False

    def forward(
        self,
        waveform: torch.Tensor,
        genre_idx: Optional[torch.Tensor] = None,
        return_params: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with automatic or manual genre selection.

        Args:
            waveform: Input audio [batch, channels, samples]
            genre_idx: Optional manual genre specification [batch]
                      If None, automatically predicts from audio
            return_params: If True, also return DSP parameters

        Returns:
            Mastered audio [batch, channels, samples]
            (optionally) DSP parameters
        """
        # If genre not provided, predict it
        if genre_idx is None:
            with torch.no_grad():
                genre_idx = self.genre_classifier(waveform)

        # Use GenreMaster with the genre
        return self.genremaster(waveform, genre_idx, return_params=return_params)

    def predict_genre(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Predict genre from audio.

        Args:
            waveform: Audio tensor [batch, channels, samples]

        Returns:
            Genre indices [batch]
        """
        with torch.no_grad():
            return self.genre_classifier(waveform)
