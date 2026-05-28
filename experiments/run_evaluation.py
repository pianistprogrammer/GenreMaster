"""Evaluation script for MUSDB18-HQ test set with baseline comparisons.

Evaluates trained GenreMaster model on MUSDB18-HQ and compares against:
- Unprocessed (pre-master baseline)
- Unconditioned GenreMaster
- Full GenreMaster (conditioned)

Usage:
    uv run python experiments/run_evaluation.py --checkpoint results/checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import json

import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.genremaster import create_genremaster_model
from features.extractor import AudioFeatureExtractor
from utils import seed_everything, get_device, load_audio, save_audio


def load_musdb_tracks(musdb_dir: Path, split: str = 'test') -> List[Dict]:
    """
    Load MUSDB18-HQ tracks.

    MUSDB18-HQ structure:
    musdb18hq/
        train/
        test/
            track1/
                mixture.wav
                vocals.wav
                bass.wav
                drums.wav
                other.wav

    Args:
        musdb_dir: Path to MUSDB18-HQ directory
        split: 'train' or 'test'

    Returns:
        List of track dictionaries with paths and metadata
    """
    split_dir = musdb_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"MUSDB split not found: {split_dir}")

    tracks = []
    for track_dir in sorted(split_dir.iterdir()):
        if not track_dir.is_dir():
            continue

        mixture_path = track_dir / "mixture.wav"
        if not mixture_path.exists():
            continue

        tracks.append({
            'name': track_dir.name,
            'mixture_path': mixture_path,
            'track_dir': track_dir,
        })

    return tracks


def evaluate_model(
    model: nn.Module,
    tracks: List[Dict],
    device: torch.device,
    sample_rate: int = 44100,
    duration: float = 30.0,
    target_lufs: float = -18.0,
) -> Dict:
    """
    Evaluate model on tracks.

    Args:
        model: GenreMaster model
        tracks: List of track dictionaries
        device: Compute device
        sample_rate: Target sample rate
        duration: Duration to evaluate (seconds)
        target_lufs: Target LUFS for pre-master simulation

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    feature_extractor = AudioFeatureExtractor(sample_rate)

    results = []

    with torch.no_grad():
        for track in tqdm(tracks, desc="Evaluating"):
            # Load mixture (mastered target)
            try:
                waveform, sr = load_audio(
                    track['mixture_path'],
                    sr=sample_rate,
                    mono=False,
                    duration=duration,
                )
            except Exception as e:
                print(f"⚠ Skipping {track['name']}: {e}")
                continue

            # Create pre-master by normalizing to target LUFS
            from data.transforms import LoudnessNormalize
            normalizer = LoudnessNormalize(target_lufs, sample_rate)
            pre_master = normalizer(waveform)

            # Add batch dimension and move to device
            pre_master_batch = pre_master.unsqueeze(0).to(device)

            # Assume Rock genre (index 0) for MUSDB (mostly rock/pop)
            genre_idx = torch.tensor([0]).to(device)

            # Forward pass
            output = model(pre_master_batch, genre_idx)
            output = output.squeeze(0).cpu()

            # Extract features
            target_features = feature_extractor.extract_all_features(waveform)
            output_features = feature_extractor.extract_all_features(output)
            premaster_features = feature_extractor.extract_all_features(pre_master)

            # Compute errors
            result = {
                'track': track['name'],
                'lufs_error': abs(output_features['lufs'] - target_features['lufs']),
                'lra_error': abs(output_features['lra'] - target_features['lra']),
                'true_peak_error': abs(output_features['true_peak'] - target_features['true_peak']),
                'spectral_tilt_error': abs(output_features['spectral_tilt'] - target_features['spectral_tilt']),
                'target_lufs': target_features['lufs'],
                'output_lufs': output_features['lufs'],
                'premaster_lufs': premaster_features['lufs'],
            }

            results.append(result)

    # Aggregate metrics
    metrics = {
        'lufs_mae': sum(r['lufs_error'] for r in results) / len(results),
        'lra_mae': sum(r['lra_error'] for r in results) / len(results),
        'true_peak_mae': sum(r['true_peak_error'] for r in results) / len(results),
        'spectral_tilt_mae': sum(r['spectral_tilt_error'] for r in results) / len(results),
        'n_tracks': len(results),
        'per_track_results': results,
    }

    return metrics


def evaluate_unprocessed_baseline(
    tracks: List[Dict],
    sample_rate: int = 44100,
    duration: float = 30.0,
    target_lufs: float = -18.0,
) -> Dict:
    """
    Evaluate unprocessed pre-master as baseline.

    Args:
        tracks: List of track dictionaries
        sample_rate: Target sample rate
        duration: Duration to evaluate
        target_lufs: Target LUFS

    Returns:
        Dictionary of metrics
    """
    feature_extractor = AudioFeatureExtractor(sample_rate)
    results = []

    for track in tqdm(tracks, desc="Baseline (unprocessed)"):
        try:
            waveform, sr = load_audio(
                track['mixture_path'],
                sr=sample_rate,
                mono=False,
                duration=duration,
            )
        except Exception:
            continue

        # Create pre-master
        from data.transforms import LoudnessNormalize
        normalizer = LoudnessNormalize(target_lufs, sample_rate)
        pre_master = normalizer(waveform)

        # Extract features
        target_features = feature_extractor.extract_all_features(waveform)
        premaster_features = feature_extractor.extract_all_features(pre_master)

        result = {
            'track': track['name'],
            'lufs_error': abs(premaster_features['lufs'] - target_features['lufs']),
            'lra_error': abs(premaster_features['lra'] - target_features['lra']),
            'true_peak_error': abs(premaster_features['true_peak'] - target_features['true_peak']),
            'spectral_tilt_error': abs(premaster_features['spectral_tilt'] - target_features['spectral_tilt']),
        }

        results.append(result)

    metrics = {
        'lufs_mae': sum(r['lufs_error'] for r in results) / len(results),
        'lra_mae': sum(r['lra_error'] for r in results) / len(results),
        'true_peak_mae': sum(r['true_peak_error'] for r in results) / len(results),
        'spectral_tilt_mae': sum(r['spectral_tilt_error'] for r in results) / len(results),
        'n_tracks': len(results),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate GenreMaster on MUSDB18-HQ")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--musdb_dir', type=str,
                        default='/Volumes/LLModels/Datasets/musdb18',
                        help='Path to MUSDB18-HQ directory')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Duration to evaluate (seconds)')
    args = parser.parse_args()

    seed_everything(42)

    print("=" * 70)
    print("GenreMaster Evaluation on MUSDB18-HQ")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MUSDB dir: {args.musdb_dir}\n")

    # Device
    device = get_device()
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MUSDB tracks
    print("Loading MUSDB18-HQ test set...")
    musdb_dir = Path(args.musdb_dir)
    tracks = load_musdb_tracks(musdb_dir, split='test')
    print(f"✓ Found {len(tracks)} test tracks\n")

    if len(tracks) == 0:
        print("⚠ No tracks found! Check MUSDB directory path.")
        return

    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    n_genres = config.get('data', {}).get('top_k_genres', 16)

    # Create conditioned model
    print("Creating GenreMaster (conditioned)...")
    model_conditioned = create_genremaster_model(
        n_genres=n_genres,
        encoder_type='lightweight',
        conditioned=True,
        sample_rate=44100,
    ).to(device)

    model_conditioned.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded\n")

    # Create unconditioned baseline
    print("Creating GenreMaster (unconditioned baseline)...")
    model_unconditioned = create_genremaster_model(
        n_genres=n_genres,
        encoder_type='lightweight',
        conditioned=False,
        sample_rate=44100,
    ).to(device)
    print("✓ Baseline model created\n")

    # Evaluate baselines
    print("=" * 70)
    print("Evaluation")
    print("=" * 70)

    # 1. Unprocessed baseline
    print("\n1. Unprocessed (Pre-master only)")
    unprocessed_metrics = evaluate_unprocessed_baseline(
        tracks,
        duration=args.duration,
    )

    print(f"   LUFS MAE: {unprocessed_metrics['lufs_mae']:.3f} dB")
    print(f"   LRA MAE: {unprocessed_metrics['lra_mae']:.3f} LU")
    print(f"   True Peak MAE: {unprocessed_metrics['true_peak_mae']:.3f} dBTP")
    print(f"   Spectral Tilt MAE: {unprocessed_metrics['spectral_tilt_mae']:.3f} dB/oct")

    # 2. Unconditioned GenreMaster
    print("\n2. GenreMaster (Unconditioned)")
    unconditioned_metrics = evaluate_model(
        model_unconditioned,
        tracks,
        device,
        duration=args.duration,
    )

    print(f"   LUFS MAE: {unconditioned_metrics['lufs_mae']:.3f} dB")
    print(f"   LRA MAE: {unconditioned_metrics['lra_mae']:.3f} LU")
    print(f"   True Peak MAE: {unconditioned_metrics['true_peak_mae']:.3f} dBTP")
    print(f"   Spectral Tilt MAE: {unconditioned_metrics['spectral_tilt_mae']:.3f} dB/oct")

    # 3. Conditioned GenreMaster (full model)
    print("\n3. GenreMaster (Full - Conditioned)")
    conditioned_metrics = evaluate_model(
        model_conditioned,
        tracks,
        device,
        duration=args.duration,
    )

    print(f"   LUFS MAE: {conditioned_metrics['lufs_mae']:.3f} dB")
    print(f"   LRA MAE: {conditioned_metrics['lra_mae']:.3f} LU")
    print(f"   True Peak MAE: {conditioned_metrics['true_peak_mae']:.3f} dBTP")
    print(f"   Spectral Tilt MAE: {conditioned_metrics['spectral_tilt_mae']:.3f} dB/oct")

    # Create comparison table
    print("\n" + "=" * 70)
    print("Comparison Table")
    print("=" * 70)

    comparison_df = pd.DataFrame([
        {
            'Model': 'Unprocessed',
            'LUFS MAE (dB)': unprocessed_metrics['lufs_mae'],
            'LRA MAE (LU)': unprocessed_metrics['lra_mae'],
            'True Peak MAE (dBTP)': unprocessed_metrics['true_peak_mae'],
            'Spectral Tilt MAE (dB/oct)': unprocessed_metrics['spectral_tilt_mae'],
        },
        {
            'Model': 'GenreMaster (Unconditioned)',
            'LUFS MAE (dB)': unconditioned_metrics['lufs_mae'],
            'LRA MAE (LU)': unconditioned_metrics['lra_mae'],
            'True Peak MAE (dBTP)': unconditioned_metrics['true_peak_mae'],
            'Spectral Tilt MAE (dB/oct)': unconditioned_metrics['spectral_tilt_mae'],
        },
        {
            'Model': 'GenreMaster (Full)',
            'LUFS MAE (dB)': conditioned_metrics['lufs_mae'],
            'LRA MAE (LU)': conditioned_metrics['lra_mae'],
            'True Peak MAE (dBTP)': conditioned_metrics['true_peak_mae'],
            'Spectral Tilt MAE (dB/oct)': conditioned_metrics['spectral_tilt_mae'],
        },
    ])

    print(comparison_df.to_string(index=False))

    # Save results
    comparison_file = output_dir / "comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison saved to: {comparison_file}")

    # Save detailed results
    detailed_file = output_dir / "detailed_results.json"
    with open(detailed_file, 'w') as f:
        json.dump({
            'unprocessed': unprocessed_metrics,
            'unconditioned': unconditioned_metrics,
            'conditioned': conditioned_metrics,
        }, f, indent=2, default=lambda x: x if not isinstance(x, float) else round(x, 6))

    print(f"✓ Detailed results saved to: {detailed_file}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
