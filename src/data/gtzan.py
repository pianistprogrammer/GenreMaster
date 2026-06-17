"""GTZAN dataset loader and preprocessing.

This module handles loading the GTZAN dataset for music genre classification.
GTZAN contains 1000 30-second audio clips across 10 genres (100 per genre).

Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import load_audio, seed_everything


# GTZAN genre list (10 genres, 100 tracks each)
GTZAN_GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# Known corrupted files in GTZAN dataset
GTZAN_CORRUPTED_FILES = [
    'jazz.00054.wav',  # Known corrupted file
]


class GTZANDataset(Dataset):
    """PyTorch Dataset for GTZAN audio tracks."""

    def __init__(
        self,
        audio_files: List[Path],
        genre_to_idx: Dict[str, int],
        sr: int = 22050,
        duration: Optional[float] = 30.0,
    ):
        """
        Initialize GTZAN dataset.

        Args:
            audio_files: List of audio file paths
            genre_to_idx: Mapping from genre name to index
            sr: Target sample rate
            duration: Duration to load in seconds (None = full track)
        """
        self.audio_files = audio_files
        self.genre_to_idx = genre_to_idx
        self.idx_to_genre = {i: g for g, i in genre_to_idx.items()}
        self.sr = sr
        self.duration = duration

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_files[idx]
        genre = audio_path.stem.split('.')[0]  # e.g., "blues.00000" -> "blues"
        genre_idx = self.genre_to_idx[genre]

        waveform, sr = load_audio(
            audio_path,
            sr=self.sr,
            mono=False,
            duration=self.duration
        )

        return {
            'waveform': waveform,
            'sample_rate': sr,
            'file_name': audio_path.name,
            'genre': genre,
            'genre_idx': genre_idx,
        }


def scan_gtzan_files(audio_dir: Path) -> List[Path]:
    """
    Scan GTZAN directory for audio files.

    Args:
        audio_dir: Path to GTZAN audio directory

    Returns:
        List of audio file paths
    """
    audio_dir = Path(audio_dir)
    audio_files = []

    # GTZAN can be in flat structure or genre subdirectories
    # Check for flat structure first: genre.XXXXX.wav
    wav_files = list(audio_dir.glob("*.wav"))
    if wav_files:
        audio_files = wav_files
    else:
        # Try subdirectory structure: genre/genre.XXXXX.wav
        for genre in GTZAN_GENRES:
            genre_dir = audio_dir / genre
            if genre_dir.exists():
                audio_files.extend(genre_dir.glob("*.wav"))

    return sorted(audio_files)


def create_gtzan_splits(
    audio_files: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """
    Create stratified train/val/test splits from GTZAN files.

    Args:
        audio_files: List of audio file paths
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing file lists
    """
    seed_everything(seed)

    # Group files by genre
    genre_files = {genre: [] for genre in GTZAN_GENRES}
    for f in audio_files:
        genre = f.stem.split('.')[0]
        if genre in genre_files:
            genre_files[genre].append(f)

    train_files, val_files, test_files = [], [], []

    for genre, files in genre_files.items():
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Shuffle files for this genre
        import random
        random.shuffle(files)

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    return {
        'train': train_files,
        'val': val_files,
        'test': test_files,
    }


def setup_gtzan(
    audio_dir: Path,
    sr: int = 22050,
    duration: Optional[float] = 30.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, GTZANDataset], Dict[str, int]]:
    """
    Setup GTZAN dataset with stratified splits.

    Args:
        audio_dir: Directory containing GTZAN audio files
        sr: Target sample rate
        duration: Duration to load in seconds
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (datasets dict, genre_to_idx mapping)
    """
    audio_dir = Path(audio_dir)

    print(f"Scanning GTZAN directory: {audio_dir}")
    audio_files = scan_gtzan_files(audio_dir)

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in {audio_dir}. "
            f"Expected .wav files named like 'genre.XXXXX.wav'"
        )

    # Verify files exist and are not corrupted
    valid_files = []
    skipped_files = []
    for f in tqdm(audio_files, desc="Verifying audio files"):
        if f.exists():
            # Skip known corrupted files
            if f.name in GTZAN_CORRUPTED_FILES:
                skipped_files.append(f.name)
                continue
            valid_files.append(f)

    print(f"✓ Found {len(valid_files)} valid audio files")
    if skipped_files:
        print(f"⚠ Skipped {len(skipped_files)} corrupted files: {skipped_files}")

    # Count by genre
    genre_counts = {}
    for f in valid_files:
        genre = f.stem.split('.')[0]
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    print("Genre distribution:")
    for genre in GTZAN_GENRES:
        count = genre_counts.get(genre, 0)
        print(f"  {genre}: {count}")

    # Create genre mapping
    genre_to_idx = {g: i for i, g in enumerate(GTZAN_GENRES)}

    # Create splits
    splits = create_gtzan_splits(
        valid_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Print split statistics
    for split_name, split_files in splits.items():
        print(f"{split_name.upper()}: {len(split_files)} tracks")

    # Create datasets
    datasets = {}
    for split_name, split_files in splits.items():
        datasets[split_name] = GTZANDataset(
            split_files,
            genre_to_idx=genre_to_idx,
            sr=sr,
            duration=duration,
        )

    print(f"\n✅ GTZAN dataset ready!")
    print(f"   Train: {len(datasets['train'])} tracks")
    print(f"   Val: {len(datasets['val'])} tracks")
    print(f"   Test: {len(datasets['test'])} tracks")
    print(f"   Genres: {len(genre_to_idx)} ({', '.join(GTZAN_GENRES)})")

    return datasets, genre_to_idx
