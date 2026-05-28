"""Test script for FMA dataset loading."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import setup_fma_medium


def test_fma_loading():
    """Test FMA Medium dataset setup."""
    # Paths
    data_root = Path("/Users/I558118/Documents/Projects/GenreMaster/data")
    audio_dir = Path("/Volumes/LLModels/Datasets/fma_medium")

    print("=" * 60)
    print("Testing FMA Medium Dataset Loading")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Audio dir: {audio_dir}")
    print()

    # Setup dataset (will download metadata if needed)
    try:
        datasets, genre_to_idx = setup_fma_medium(
            data_root=data_root,
            audio_dir=audio_dir,
            top_k_genres=16,
            samples_per_genre=1500,
        )

        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)

        # Print genre mapping
        print(f"\nGenre mapping ({len(genre_to_idx)} genres):")
        for genre_id, idx in sorted(genre_to_idx.items(), key=lambda x: x[1])[:5]:
            print(f"  Genre {genre_id} → Index {idx}")
        print("  ...")

        # Test loading a sample from each split
        print("\nLoading sample from each split:")
        for split_name, dataset in datasets.items():
            sample = dataset[0]
            print(f"\n{split_name.upper()} sample:")
            print(f"  Track ID: {sample['track_id']}")
            print(f"  Waveform shape: {sample['waveform'].shape}")
            print(f"  Sample rate: {sample['sample_rate']}")
            print(f"  Genre ID: {sample['genre_id']} (index: {sample['genre_idx']})")

        print("\n" + "=" * 60)
        print("✅ FMA dataset loading test PASSED!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ FMA dataset loading test FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    test_fma_loading()
