"""
Validate all audio files in FMA dataset and identify corrupted ones.
Uses parallel processing for I/O-bound file validation.
"""
import os
import sys
from pathlib import Path
import torchaudio
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def validate_audio_file(audio_path, sr=44100):
    """Check if audio file can be loaded without errors."""
    try:
        waveform, sr_loaded = torchaudio.load(str(audio_path))
        # Check if audio is mostly silence
        max_val = waveform.abs().max()
        if max_val < 1e-6:  # Essentially silent
            return False, "silent"
        return True, None
    except Exception as e:
        return False, str(type(e).__name__)

def main():
    audio_dir = Path("C:/Users/jerem/Documents/Datasets/fma_small")
    metadata_path = Path("C:/Users/jerem/Documents/Datasets/fma_metadata/tracks.csv")
    
    # Load metadata
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    
    # Filter by subset (small)
    subset = tracks['set', 'subset'] <= 'small'
    tracks = tracks[subset]
    
    # Filter tracks with genre
    tracks = tracks[tracks['track', 'genre_top'].notna()]
    
    # Build list of (track_id, audio_path) tuples
    files_to_validate = []
    for track_id in tracks.index:
        tid_str = f"{track_id:06d}"
        audio_path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
        if audio_path.exists():
            files_to_validate.append((track_id, audio_path))
    
    corrupted_files = []
    valid_count = 0
    num_workers = os.cpu_count() or 4
    
    print(f"Validating {len(files_to_validate)} audio files with {num_workers} workers...")
    
    # Parallel validation using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all validation tasks
        future_to_track = {
            executor.submit(validate_audio_file, path): (track_id, path)
            for track_id, path in files_to_validate
        }
        
        # Process completed tasks with progress bar
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
        # Save corrupted file list
        df_corrupted = pd.DataFrame(corrupted_files)
        output_path = Path("results/corrupted_files.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_corrupted.to_csv(output_path, index=False)
        print(f"\nCorrupted files saved to: {output_path}")
        
        # Print summary by error type
        print("\nError breakdown:")
        for error_type in df_corrupted['error'].unique():
            count = (df_corrupted['error'] == error_type).sum()
            print(f"  {error_type}: {count} files")

if __name__ == "__main__":
    main()
