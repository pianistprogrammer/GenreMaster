"""Test GTZAN files for corruption."""
import sys
from pathlib import Path
import torchaudio
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.gtzan import setup_gtzan

# Load GTZAN
datasets, genre_to_idx = setup_gtzan(
    data_root=Path("data"),
    audio_dir=Path("C:/Users/jerem/Documents/Datasets/GTZAN/genres_original"),
    num_genres=10,
)

train_dataset = datasets['train']

# Test all files in training set
print(f"Testing all {len(train_dataset)} files in training set:")
print("=" * 70)

problematic_files = []

for idx in range(len(train_dataset)):
    try:
        sample = train_dataset[idx]
        waveform = sample['waveform']
        path = sample.get('path', 'unknown')
        
        # Check for NaN/inf/silent
        has_nan = torch.isnan(waveform).any()
        has_inf = torch.isinf(waveform).any()
        mean_val = waveform.mean().item()
        std_val = waveform.std().item()
        
        # Flag problems
        if has_nan or has_inf or std_val < 0.0001:
            print(f"[{idx:3d}] ✗ PROBLEM | mean={mean_val:.6f}, std={std_val:.6f} | {path}")
            problematic_files.append({
                'idx': idx,
                'path': path,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'std': std_val,
                'mean': mean_val
            })
        elif idx % 100 == 0:
            print(f"[{idx:3d}] ✓ Checked {idx} files...")
        
    except Exception as e:
        print(f"[{idx:3d}] ✗ ERROR: {str(e)[:60]}")
        problematic_files.append({
            'idx': idx,
            'path': '?',
            'error': str(e)[:60]
        })

print("=" * 70)
if problematic_files:
    print(f"\nFound {len(problematic_files)} problematic files:")
    for p in problematic_files[:10]:  # Show first 10
        print(f"  Index {p['idx']}: {p.get('path', '?')}")
        if 'error' in p:
            print(f"    Error: {p['error']}")
        else:
            print(f"    NaN={p.get('has_nan')}, Inf={p.get('has_inf')}, Std={p.get('std', '?'):.6f}")
else:
    print("\n✓ All files OK!")
