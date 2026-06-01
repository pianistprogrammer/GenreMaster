from pathlib import Path
import torchaudio
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

def validate_audio_file_strict(audio_path, sr=44100, duration=30.0, tolerance=0.1):
    try:
        waveform, sr_loaded = torchaudio.load(str(audio_path))
        expected_length = int(sr * duration)
        actual_length = waveform.shape[1]
        
        if actual_length < expected_length * (1 - tolerance):
            return False, "too_short"
        if actual_length > expected_length * (1 + tolerance):
            return False, "too_long"
        
        max_val = waveform.abs().max()
        if max_val < 1e-6:
            return False, "silent"
        
        if max_val > 1.5:
            return False, "clipped"
        
        if not (waveform.isfinite().all()):
            return False, "has_nan_or_inf"
        
        return True, None
    except Exception as e:
        return False, str(type(e).__name__)

audio_dir = Path("C:/Users/jerem/Documents/Datasets/fma_small")
metadata_path = Path("C:/Users/jerem/Documents/Datasets/fma_metadata/tracks.csv")

print("Loading metadata...")
tracks = pd.read_csv(str(metadata_path), index_col=0, header=[0, 1])
subset = tracks['set', 'subset'] <= 'small'
tracks = tracks[subset]
tracks = tracks[tracks['track', 'genre_top'].notna()]
top_genres = tracks['track', 'genre_top'].value_counts().nlargest(8).index
tracks = tracks[tracks['track', 'genre_top'].isin(top_genres)]
print(f"Filtered to {len(tracks)} tracks")

files_to_validate = []
for track_id in tracks.index:
    tid_str = f"{track_id:06d}"
    audio_path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
    if audio_path.exists():
        files_to_validate.append((track_id, audio_path))

print(f"Found {len(files_to_validate)} files")
print(f"Validating with {os.cpu_count()} workers...")

corrupted_files = []
valid_count = 0
num_workers = os.cpu_count() or 4

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_track = {
        executor.submit(validate_audio_file_strict, path): (track_id, path)
        for track_id, path in files_to_validate
    }
    
    with tqdm(total=len(files_to_validate), desc="Scanning") as pbar:
        for future in as_completed(future_to_track):
            track_id, path = future_to_track[future]
            is_valid, error_type = future.result()
            
            if not is_valid:
                corrupted_files.append(track_id)
            else:
                valid_count += 1
            
            pbar.update(1)

print(f"\n✓ Valid files: {valid_count}")
print(f"✗ Problematic files: {len(corrupted_files)}")
if corrupted_files:
    problematic_ids = sorted(corrupted_files)
    print(f"Problematic track IDs: {problematic_ids}")
