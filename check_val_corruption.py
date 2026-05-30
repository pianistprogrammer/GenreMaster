"""
Find all corrupted files in validation set specifically
"""
from pathlib import Path
import torchaudio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def test_load_audio(audio_path):
    """Try to load audio and return status."""
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        # Check for silence
        if waveform.abs().max() < 1e-6:
            return False, "silent"
        # Check for NaN/Inf
        if not waveform.isfinite().all():
            return False, "has_nan_or_inf"
        return True, None
    except Exception as e:
        return False, str(type(e).__name__)

audio_dir = Path("C:/Users/jerem/Documents/Datasets/fma_small")
metadata_path = Path("C:/Users/jerem/Documents/Datasets/fma_metadata/tracks.csv")

print("Loading metadata...")
tracks = pd.read_csv(str(metadata_path), index_col=0, header=[0, 1])

# Filter for validation set
subset = tracks['set', 'subset'] <= 'small'
tracks = tracks[subset]
tracks = tracks[tracks['track', 'genre_top'].notna()]
top_genres = tracks['track', 'genre_top'].value_counts().nlargest(8).index
tracks = tracks[tracks['track', 'genre_top'].isin(top_genres)]

val_ids = tracks[tracks['set', 'split'] == 'validation'].index.tolist()
print(f"Found {len(val_ids)} validation tracks")

# Test each validation file
corrupted_val = []
for track_id in val_ids:
    tid_str = f"{track_id:06d}"
    audio_path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
    if audio_path.exists():
        is_valid, error = test_load_audio(audio_path)
        if not is_valid:
            corrupted_val.append((track_id, error))

print(f"\n✗ Corrupted validation files: {len(corrupted_val)}")
if corrupted_val:
    for track_id, error in corrupted_val:
        print(f"  {track_id}: {error}")
