import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

FMA_SMALL_GENRES = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock',
]

# Known corrupted FMA track IDs
FMA_CORRUPTED = {98565, 98567, 98569, 99134, 108925, 133297}


class FMASmallDataset(Dataset):
    def __init__(
        self,
        track_ids: List[int],
        audio_dir: Path,
        genre_to_idx: Dict[str, int],
        metadata: pd.DataFrame,
        sample_rate: int = 22050,
        duration: float = 30.0,
        cache_dir: Optional[Path] = None,
    ):
        self.audio_dir = Path(audio_dir)
        self.genre_to_idx = genre_to_idx
        self.idx_to_genre = {v: k for k, v in genre_to_idx.items()}
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.track_ids = track_ids
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.track_ids)

    def _audio_path(self, track_id: int) -> Path:
        tid = f"{track_id:06d}"
        return self.audio_dir / tid[:3] / f"{tid}.mp3"

    def __getitem__(self, idx: int) -> Dict:
        track_id = self.track_ids[idx]
        genre_str = self.metadata.loc[track_id, ('track', 'genre_top')]
        genre_idx = self.genre_to_idx[genre_str]

        try:
            # Use preprocessed cache if available
            if self.cache_dir is not None:
                cache_path = self.cache_dir / f"{track_id}.pt"
                if cache_path.exists():
                    waveform = torch.load(cache_path, weights_only=True)
                    return {
                        'waveform': waveform,
                        'genre_idx': genre_idx,
                        'genre': genre_str,
                        'track_id': track_id,
                        'sample_rate': self.sample_rate,
                    }

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                waveform, sr = torchaudio.load(str(self._audio_path(track_id)))

            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

            # Ensure mono (FMA is stereo but model encoder averages channels)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Trim / pad to exact length
            if waveform.shape[1] >= self.target_length:
                # Take centre crop
                start = (waveform.shape[1] - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[1]))

        except Exception:
            waveform = torch.zeros((1, self.target_length))

        return {
            'waveform': waveform,
            'genre_idx': genre_idx,
            'genre': genre_str,
            'track_id': track_id,
            'sample_rate': self.sample_rate,
        }


def setup_fma_small(
    audio_dir: str,
    metadata_csv: str,
    sample_rate: int = 22050,
    duration: float = 30.0,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, FMASmallDataset], Dict[str, int]]:
    """
    Load FMA-small using the official train/validation/test splits from the
    FMA metadata CSV. Returns datasets dict and genre_to_idx mapping.
    """
    audio_dir = Path(audio_dir)
    tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])

    # Keep only fma_small subset with a top-level genre
    small = tracks[tracks['set', 'subset'] == 'small'].copy()
    small = small[small['track', 'genre_top'].isin(FMA_SMALL_GENRES)]

    # Remove known corrupted files and check physical presence
    valid_ids = []
    for tid in small.index:
        if tid in FMA_CORRUPTED:
            continue
        tid_str = f"{tid:06d}"
        if (audio_dir / tid_str[:3] / f"{tid_str}.mp3").exists():
            valid_ids.append(tid)
    small = small.loc[valid_ids]

    genre_to_idx = {g: i for i, g in enumerate(FMA_SMALL_GENRES)}

    split_map = {'training': 'train', 'validation': 'val', 'test': 'test'}
    datasets = {}
    for fma_split, split_name in split_map.items():
        ids = small[small['set', 'split'] == fma_split].index.tolist()
        datasets[split_name] = FMASmallDataset(
            ids, audio_dir, genre_to_idx, small, sample_rate, duration,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

    return datasets, genre_to_idx
