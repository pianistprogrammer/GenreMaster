"""Extract mastering signatures from FMA dataset.

This script processes all FMA Medium tracks and extracts mastering-relevant
features, saving results to a CSV file with checkpoint/resume functionality.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Set
import sys

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import load_fma_metadata
from features.extractor import extract_features_from_file
from utils import seed_everything


class SignatureExtractor:
    """Extract mastering signatures with checkpoint/resume support."""

    def __init__(self, output_dir: Path):
        """
        Initialize signature extractor.

        Args:
            output_dir: Directory for output files and checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "extraction_checkpoint.json"
        self.results_file = self.output_dir / "fma_signatures.csv"

        self.completed: Set[int] = self._load_checkpoint()
        self.results: List[Dict] = []

    def _load_checkpoint(self) -> Set[int]:
        """Load checkpoint of completed track IDs."""
        if not self.checkpoint_file.exists():
            return set()

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get("completed_tracks", []))
        except Exception as e:
            print(f"⚠ Warning: Could not load checkpoint: {e}")
            return set()

    def _save_checkpoint(self, track_id: int, result: Dict):
        """Save checkpoint after each successful extraction."""
        self.completed.add(track_id)
        self.results.append(result)

        # Update checkpoint file
        checkpoint = {
            "completed_tracks": list(self.completed),
            "n_completed": len(self.completed),
            "last_updated": time.time(),
            "last_track_id": track_id,
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Append to results CSV (incremental save)
        df = pd.DataFrame([result])
        if not self.results_file.exists():
            df.to_csv(self.results_file, index=False)
        else:
            df.to_csv(self.results_file, mode='a', header=False, index=False)

    def extract_signatures(
        self,
        tracks_df: pd.DataFrame,
        audio_dir: Path,
        sample_rate: int = 44100,
        duration: float = 30.0,
    ):
        """
        Extract mastering signatures from FMA tracks.

        Args:
            tracks_df: DataFrame with track metadata
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            duration: Duration to analyze (seconds)
        """
        # Filter for tracks with valid audio files
        track_ids = tracks_df.index.tolist()

        # Remove already-completed tracks
        remaining_ids = [tid for tid in track_ids if tid not in self.completed]

        print(f"\n{'=' * 60}")
        print("FMA Mastering Signature Extraction")
        print(f"{'=' * 60}")
        print(f"Total tracks: {len(track_ids)}")
        print(f"Already completed: {len(self.completed)}")
        print(f"Remaining: {len(remaining_ids)}")
        print(f"Output: {self.results_file}")
        print(f"Checkpoint: {self.checkpoint_file}")
        print(f"{'=' * 60}\n")

        if len(remaining_ids) == 0:
            print("✅ All tracks already processed!")
            return

        # Extract features for remaining tracks
        failed_tracks = []

        for track_id in tqdm(remaining_ids, desc="Extracting signatures"):
            audio_path = audio_dir / f"{track_id:06d}.mp3"

            if not audio_path.exists():
                continue

            try:
                # Extract features
                features = extract_features_from_file(str(audio_path), sample_rate)

                # Get genre from metadata
                if isinstance(tracks_df.columns, pd.MultiIndex):
                    genre = tracks_df.loc[track_id, ('track', 'genre_top')]
                else:
                    genre = tracks_df.loc[track_id, 'genre_top']

                # Build result record
                result = {
                    'track_id': track_id,
                    'genre': genre,
                    'lufs': features['lufs'],
                    'lra': features['lra'],
                    'true_peak': features['true_peak'],
                    'spectral_tilt': features['spectral_tilt'],
                    'ms_ratio': features['ms_ratio'],
                    'dr14': features['dr14'],
                }

                # Save checkpoint
                self._save_checkpoint(track_id, result)

            except Exception as e:
                failed_tracks.append((track_id, str(e)))
                continue

        print(f"\n{'=' * 60}")
        print("Extraction Complete!")
        print(f"{'=' * 60}")
        print(f"Successfully extracted: {len(self.completed)}")
        print(f"Failed: {len(failed_tracks)}")

        if failed_tracks:
            print(f"\nFailed tracks:")
            for track_id, error in failed_tracks[:10]:
                print(f"  Track {track_id}: {error}")
            if len(failed_tracks) > 10:
                print(f"  ... and {len(failed_tracks) - 10} more")

        print(f"\n✅ Results saved to: {self.results_file}")
        print(f"{'=' * 60}\n")

    def compute_genre_statistics(self):
        """Compute per-genre statistics from extracted signatures."""
        if not self.results_file.exists():
            print("⚠ No results file found. Run extraction first.")
            return

        df = pd.read_csv(self.results_file)

        print(f"\n{'=' * 60}")
        print("Genre-Specific Mastering Statistics")
        print(f"{'=' * 60}\n")

        # Group by genre
        genre_stats = df.groupby('genre').agg({
            'lufs': ['mean', 'std', 'min', 'max'],
            'lra': ['mean', 'std', 'min', 'max'],
            'true_peak': ['mean', 'std', 'min', 'max'],
            'spectral_tilt': ['mean', 'std', 'min', 'max'],
            'ms_ratio': ['mean', 'std', 'min', 'max'],
            'dr14': ['mean', 'std', 'min', 'max'],
        }).round(2)

        # Count per genre
        genre_counts = df['genre'].value_counts().sort_index()

        print("Track counts per genre:")
        print(genre_counts)
        print()

        print("LUFS (Integrated Loudness) by Genre:")
        print(genre_stats['lufs'])
        print()

        print("LRA (Loudness Range) by Genre:")
        print(genre_stats['lra'])
        print()

        print("DR14 (Dynamic Range) by Genre:")
        print(genre_stats['dr14'])
        print()

        # Save genre statistics
        stats_file = self.output_dir / "genre_statistics.csv"
        genre_stats.to_csv(stats_file)
        print(f"✅ Genre statistics saved to: {stats_file}")

        # Save track counts
        counts_file = self.output_dir / "genre_counts.csv"
        genre_counts.to_csv(counts_file, header=['count'])
        print(f"✅ Genre counts saved to: {counts_file}")

        print(f"{'=' * 60}\n")


def main():
    """Main execution function."""
    seed_everything(42)

    # Paths
    data_root = Path("/Users/I558118/Documents/Projects/GenreMaster/data")
    audio_dir = Path("/Volumes/LLModels/Datasets/fma_medium")
    output_dir = data_root / "processed"

    # Load FMA metadata
    metadata_dir = data_root / "fma_metadata"
    if not metadata_dir.exists():
        print("⚠ FMA metadata not found. Run test_fma.py first to download.")
        return

    tracks, genres = load_fma_metadata(metadata_dir)

    # Filter for FMA Medium tracks only
    if isinstance(tracks.columns, pd.MultiIndex):
        medium_mask = tracks[('set', 'subset')] == 'medium'
    else:
        medium_mask = tracks['subset'] == 'medium'

    tracks_medium = tracks[medium_mask]

    print(f"✓ Loaded {len(tracks_medium)} FMA Medium tracks")

    # Initialize extractor
    extractor = SignatureExtractor(output_dir)

    # Extract signatures (with resume support)
    extractor.extract_signatures(
        tracks_medium,
        audio_dir,
        sample_rate=44100,
        duration=30.0,  # Analyze 30 seconds
    )

    # Compute genre statistics
    extractor.compute_genre_statistics()


if __name__ == "__main__":
    main()
