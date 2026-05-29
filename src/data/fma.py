import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

class FMADataset(Dataset):
    def __init__(self, track_ids, audio_dir, genre_to_idx, metadata, sample_rate=44100, duration=30.0, transform=None):
        self.track_ids = track_ids
        self.audio_dir = Path(audio_dir)
        self.genre_to_idx = genre_to_idx
        self.metadata = metadata
        self.idx_to_genre = {v: k for k, v in genre_to_idx.items()}
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.transform = transform

    def __len__(self):
        return len(self.track_ids)

    def _get_audio_path(self, track_id):
        tid_str = f"{track_id:06d}"
        return self.audio_dir / tid_str[:3] / f"{tid_str}.mp3"

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        audio_path = self._get_audio_path(track_id)
        
        # Get actual genre from metadata
        genre_str = self.metadata.loc[track_id, ('track', 'genre_top')]
        genre_idx = self.genre_to_idx[genre_str]
        
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Stereo if needed
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
                
            # Trim / pad
            if waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]
            elif waveform.shape[1] < self.target_length:
                pad_len = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        except Exception:
            # Fallback to zeros
            waveform = torch.zeros((2, self.target_length))
            
        if self.transform:
            waveform = self.transform(waveform)
            
        return {
            'waveform': waveform,
            'genre_id': genre_str,
            'genre_idx': genre_idx,
            'track_id': track_id,
            'sample_rate': self.sample_rate
        }

def setup_fma_medium(data_root, audio_dir, top_k_genres=16, samples_per_genre=1500):
    """
    Sets up the FMA small/medium dataset using the actual metadata.
    """
    audio_dir = Path(audio_dir)
    # Use the parent directory of datasets if absolute data root provided isn't right
    metadata_path = Path("C:/Users/jerem/Documents/Datasets/fma_metadata/tracks.csv")
    
    # Load metadata
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    
    # Filter by subset (small)
    subset = tracks['set', 'subset'] <= 'small' # Filter for small subset tracks
    tracks = tracks[subset]
    
    # Filter tracks that have a genre_top and keep only top_k_genres
    tracks = tracks[tracks['track', 'genre_top'].notna()]
    top_genres = tracks['track', 'genre_top'].value_counts().nlargest(top_k_genres).index
    tracks = tracks[tracks['track', 'genre_top'].isin(top_genres)]
    
    genre_to_idx = {genre: i for i, genre in enumerate(top_genres)}
    
    # Check physical presence
    valid_track_ids = []
    for track_id in tracks.index:
        tid_str = f"{track_id:06d}"
        path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
        if path.exists():
            valid_track_ids.append(track_id)
            
    tracks = tracks.loc[valid_track_ids]
    
    # Split using FMA's official splits
    train_ids = tracks[tracks['set', 'split'] == 'training'].index.tolist()
    val_ids = tracks[tracks['set', 'split'] == 'validation'].index.tolist()
    test_ids = tracks[tracks['set', 'split'] == 'test'].index.tolist()
    
    datasets = {
        'train': FMADataset(train_ids, audio_dir, genre_to_idx, tracks),
        'val': FMADataset(val_ids, audio_dir, genre_to_idx, tracks),
        'test': FMADataset(test_ids, audio_dir, genre_to_idx, tracks)
    }
    
    return datasets, genre_to_idx
