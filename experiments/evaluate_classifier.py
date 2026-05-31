#!/usr/bin/env python
"""Evaluate genre classifier with segment voting and TTA.

Usage:
    # Evaluate with segment voting + TTA
    uv run python experiments/evaluate_classifier.py

    # Evaluate without TTA (faster)
    uv run python experiments/evaluate_classifier.py --no-tta

    # Custom checkpoint
    uv run python experiments/evaluate_classifier.py --checkpoint results/my_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.gtzan import setup_gtzan, GTZAN_GENRES
from models.resnet_genre_classifier import ResNetGenreClassifier
from utils import seed_everything, get_device


def collate_fn(batch):
    """Custom collate function."""
    min_length = min(item['waveform'].shape[1] for item in batch)
    waveforms = torch.stack([item['waveform'][:, :min_length] for item in batch])
    genre_indices = torch.tensor([item['genre_idx'] for item in batch])
    return {'waveform': waveforms, 'genre_idx': genre_indices}


def evaluate(
    model,
    test_loader,
    device,
    use_segments: bool = True,
    use_tta: bool = True,
    n_tta_augments: int = 5,
    segment_seconds: float = 3.0,
    hop_seconds: float = 1.5,
):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    desc = "Evaluating"
    if use_tta:
        desc += " (TTA)"
    if use_segments:
        desc += " (Segments)"

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=desc):
            waveform = batch['waveform'].to(device)
            labels = batch['genre_idx']

            if use_tta:
                preds, probs = model.predict_with_tta(
                    waveform,
                    n_augments=n_tta_augments,
                    use_segments=use_segments,
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                )
            elif use_segments:
                preds, probs = model.predict_with_segments(
                    waveform,
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                )
            else:
                logits = model(waveform)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / len(all_labels)

    return accuracy, all_preds, all_labels, all_probs


def print_confusion_matrix(preds, labels, genres):
    """Print confusion matrix."""
    import numpy as np

    n_classes = len(genres)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for p, l in zip(preds, labels):
        cm[l][p] += 1

    print("\nConfusion Matrix:")
    print("-" * 80)

    # Header
    header = "          " + " ".join([f"{g[:5]:>5}" for g in genres])
    print(header)

    # Rows
    for i, genre in enumerate(genres):
        row = f"{genre[:8]:>8}: " + " ".join([f"{cm[i][j]:>5}" for j in range(n_classes)])
        print(row)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, genre in enumerate(genres):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = 100.0 * correct / total if total > 0 else 0
        print(f"  {genre}: {acc:.1f}% ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Genre Classifier")
    parser.add_argument('--checkpoint', type=str,
                        default='results/genre_classifier_gtzan_resnet_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/gtzan.yaml',
                        help='Path to config file')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--no-segments', action='store_true',
                        help='Disable segment voting')
    parser.add_argument('--n-tta', type=int, default=5,
                        help='Number of TTA augmentations')
    parser.add_argument('--segment-sec', type=float, default=3.0,
                        help='Segment duration in seconds')
    parser.add_argument('--hop-sec', type=float, default=1.5,
                        help='Hop duration in seconds')
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Genre Classifier Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"TTA: {'Enabled' if not args.no_tta else 'Disabled'}")
    print(f"Segments: {'Enabled' if not args.no_segments else 'Disabled'}")

    # Load dataset (test split)
    print("\nLoading GTZAN test set...")
    datasets, genre_to_idx = setup_gtzan(
        audio_dir=Path(config['data']['audio_dir']),
        sr=config['data'].get('sample_rate', 22050),
        duration=config['data'].get('duration', 30.0),
    )

    test_loader = DataLoader(
        datasets['test'],
        batch_size=1,  # Process one at a time for TTA
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"✓ Test samples: {len(datasets['test'])}")

    # Load model
    print("\nLoading model...")
    model = ResNetGenreClassifier(
        n_genres=len(genre_to_idx),
        n_mels=config['model'].get('n_mels', 128),
        n_fft=config['model'].get('n_fft', 2048),
        hop_length=config['model'].get('hop_length', 512),
        sample_rate=config['data'].get('sample_rate', 22050),
        pretrained=False,  # Will load weights
        spec_augment=False,  # No augment during eval
    )

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Model loaded")

    # Evaluate with different settings
    print("\n" + "=" * 70)

    # 1. Basic (no segments, no TTA)
    print("\n[1] Basic evaluation (no segments, no TTA):")
    acc_basic, _, _, _ = evaluate(
        model, test_loader, device,
        use_segments=False, use_tta=False
    )
    print(f"    Accuracy: {acc_basic:.2f}%")

    # 2. With segments only
    if not args.no_segments:
        print(f"\n[2] With segment voting ({args.segment_sec}s segments, {args.hop_sec}s hop):")
        acc_seg, _, _, _ = evaluate(
            model, test_loader, device,
            use_segments=True, use_tta=False,
            segment_seconds=args.segment_sec,
            hop_seconds=args.hop_sec,
        )
        print(f"    Accuracy: {acc_seg:.2f}%")

    # 3. With TTA only
    if not args.no_tta:
        print(f"\n[3] With TTA ({args.n_tta} augmentations):")
        acc_tta, _, _, _ = evaluate(
            model, test_loader, device,
            use_segments=False, use_tta=True,
            n_tta_augments=args.n_tta,
        )
        print(f"    Accuracy: {acc_tta:.2f}%")

    # 4. Full (segments + TTA)
    if not args.no_segments and not args.no_tta:
        print(f"\n[4] Full (segments + TTA):")
        acc_full, preds, labels, probs = evaluate(
            model, test_loader, device,
            use_segments=True, use_tta=True,
            n_tta_augments=args.n_tta,
            segment_seconds=args.segment_sec,
            hop_seconds=args.hop_sec,
        )
        print(f"    Accuracy: {acc_full:.2f}%")

        # Print confusion matrix for best result
        print_confusion_matrix(preds, labels, GTZAN_GENRES)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Basic:         {acc_basic:.2f}%")
    if not args.no_segments:
        print(f"  + Segments:    {acc_seg:.2f}%")
    if not args.no_tta:
        print(f"  + TTA:         {acc_tta:.2f}%")
    if not args.no_segments and not args.no_tta:
        print(f"  + Both:        {acc_full:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
