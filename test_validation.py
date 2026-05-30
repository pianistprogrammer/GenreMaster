from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchaudio
import os
from tqdm import tqdm

def validate_audio_file(audio_path, sr=44100):
    try:
        waveform, sr_loaded = torchaudio.load(str(audio_path))
        max_val = waveform.abs().max()
        if max_val < 1e-6:
            return False, "silent"
        return True, None
    except Exception as e:
        return False, str(type(e).__name__)

print("Loading metadata...")
tracks = pd.read_csv("C:/Users/jerem/Documents/Datasets/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
print(f"Loaded {len(tracks)} tracks")

subset = tracks['set', 'subset'] <= 'small'
tracks = tracks[subset]
print(f"Filtered to {len(tracks)} tracks (small subset)")

tracks = tracks[tracks['track', 'genre_top'].notna()]
print(f"Filtered to {len(tracks)} tracks with genre")

audio_dir = Path("C:/Users/jerem/Documents/Datasets/fma_small")
files_to_validate = []
for track_id in tracks.index:
    tid_str = f"{track_id:06d}"
    audio_path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
    if audio_path.exists():
        files_to_validate.append((track_id, audio_path))

print(f"Found {len(files_to_validate)} files to validate")

corrupted_files = []
valid_count = 0
num_workers = os.cpu_count() or 4

print(f"\nValidating with {num_workers} parallel workers...")

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_track = {
        executor.submit(validate_audio_file, path): (track_id, path)
        for track_id, path in files_to_validate
    }
    
    with tqdm(total=len(files_to_validate), desc="Scanning files") as pbar:
        for future in as_completed(future_to_track):
            track_id, path = future_to_track[future]
            is_valid, error_type = future.result()
            
            if not is_valid:
                corrupted_files.append({
                    'track_id': track_id,
                    'path': str(path),
                    'error': error_type
                })
            else:
                valid_count += 1
            
            pbar.update(1)

print(f"\n✓ Valid files: {valid_count}")
print(f"✗ Corrupted files: {len(corrupted_files)}")

if corrupted_files:
    df_corrupted = pd.DataFrame(corrupted_files)
    print("\nError breakdown:")
    for error_type in df_corrupted['error'].unique():
        count = (df_corrupted['error'] == error_type).sum()
        print(f"  {error_type}: {count} files")
