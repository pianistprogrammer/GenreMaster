#!/usr/bin/env python
"""Evaluate trained GenreMaster mastering model.

Usage:
    uv run python experiments/evaluate_mastering.py
    uv run python experiments/evaluate_mastering.py --checkpoint results/checkpoints/best_model.pt
    uv run python experiments/evaluate_mastering.py --save-audio  # Save example outputs
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*stft.*')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.gtzan import setup_gtzan, GTZAN_GENRES
from data.transforms import create_premaster_transforms
from losses import create_loss_function
from models.genremaster import create_genremaster_model
from utils import seed_everything, get_device


def collate_fn(batch):
    """Custom collate function for variable length audio."""
    processed = []
    for item in batch:
        waveform = item['waveform']
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        processed.append(waveform)

    min_len = min(w.shape[1] for w in processed)
    return {
        'waveform': torch.stack([w[:, :min_len] for w in processed]),
        'genre_idx': torch.tensor([item['genre_idx'] for item in batch]),
        'genre': [item['genre'] for item in batch],
        'file_path': [item.get('file_path', 'unknown') for item in batch],
    }


def compute_metrics(pred, target, sample_rate=22050):
    """Compute audio quality metrics."""
    metrics = {}

    # MSE
    metrics['mse'] = F.mse_loss(pred, target).item()

    # L1
    metrics['l1'] = F.l1_loss(pred, target).item()

    # SNR (Signal-to-Noise Ratio)
    noise = pred - target
    signal_power = (target ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power > 1e-10:
        metrics['snr_db'] = 10 * torch.log10(signal_power / noise_power).item()
    else:
        metrics['snr_db'] = 100.0  # Very high SNR

    # Correlation
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    correlation = (pred_centered * target_centered).sum() / (
        pred_centered.norm() * target_centered.norm() + 1e-8
    )
    metrics['correlation'] = correlation.item()

    # RMS difference (loudness proxy)
    pred_rms = torch.sqrt((pred ** 2).mean()).item()
    target_rms = torch.sqrt((target ** 2).mean()).item()
    metrics['rms_pred'] = pred_rms
    metrics['rms_target'] = target_rms
    metrics['rms_diff_db'] = 20 * torch.log10(
        torch.tensor(max(pred_rms, 1e-8) / max(target_rms, 1e-8))
    ).item()

    return metrics


def evaluate_model(
    model,
    test_loader,
    criterion,
    transforms,
    device,
    save_audio=False,
    output_dir=None,
    num_examples=5,
):
    """Evaluate model on test set."""
    model.eval()
    train_tf, val_tf = transforms

    all_losses = []
    all_metrics = []
    per_genre_losses = {genre: [] for genre in GTZAN_GENRES}

    saved_examples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)
            genres = batch['genre']

            # Apply degradation transform to create pre-master
            pre_masters, targets = [], []
            for i in range(waveform.shape[0]):
                pm, tgt = val_tf(waveform[i])
                pre_masters.append(pm)
                targets.append(tgt)

            pre_master_batch = torch.stack(pre_masters).to(device)
            target_batch = torch.stack(targets).to(device)

            # Forward pass
            output = model(pre_master_batch, genre_idx)

            # Compute loss
            losses = criterion(output, target_batch, return_components=True)
            all_losses.append({k: v.item() for k, v in losses.items()})

            # Compute metrics per sample
            for i in range(output.shape[0]):
                metrics = compute_metrics(output[i], target_batch[i])
                metrics['genre'] = genres[i]
                all_metrics.append(metrics)
                per_genre_losses[genres[i]].append(losses['total'].item())

            # Save audio examples
            if save_audio and saved_examples < num_examples:
                for i in range(min(output.shape[0], num_examples - saved_examples)):
                    save_audio_example(
                        pre_master_batch[i].cpu(),
                        output[i].cpu(),
                        target_batch[i].cpu(),
                        genres[i],
                        output_dir,
                        saved_examples,
                    )
                    saved_examples += 1

    return all_losses, all_metrics, per_genre_losses


def save_audio_example(pre_master, output, target, genre, output_dir, idx):
    """Save audio example for listening."""
    import scipy.io.wavfile as wavfile
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 22050

    # Normalize to prevent clipping
    def normalize(x):
        x_np = x.numpy().squeeze()
        peak = np.abs(x_np).max()
        if peak > 0.99:
            x_np = x_np * 0.99 / peak
        # Convert to int16 for wav file
        return (x_np * 32767).astype(np.int16)

    wavfile.write(
        output_dir / f"example_{idx}_{genre}_input.wav",
        sample_rate, normalize(pre_master)
    )
    wavfile.write(
        output_dir / f"example_{idx}_{genre}_output.wav",
        sample_rate, normalize(output)
    )
    wavfile.write(
        output_dir / f"example_{idx}_{genre}_target.wav",
        sample_rate, normalize(target)
    )


def print_results(all_losses, all_metrics, per_genre_losses):
    """Print evaluation results."""
    import numpy as np

    print("\n" + "=" * 70)
    print("GENREMASTER EVALUATION RESULTS")
    print("=" * 70)

    # Overall loss
    avg_loss = np.mean([l['total'] for l in all_losses])
    avg_loudness = np.mean([l['loudness'] for l in all_losses])
    avg_spectral = np.mean([l['spectral'] for l in all_losses])
    avg_dynamic = np.mean([l['dynamic'] for l in all_losses])
    avg_perceptual = np.mean([l['perceptual'] for l in all_losses])

    print(f"\n📊 OVERALL LOSS (Test Set)")
    print(f"   Total Loss:      {avg_loss:.4f}")
    print(f"   Loudness Loss:   {avg_loudness:.4f}")
    print(f"   Spectral Loss:   {avg_spectral:.4f}")
    print(f"   Dynamic Loss:    {avg_dynamic:.4f}")
    print(f"   Perceptual Loss: {avg_perceptual:.4f}")

    # Audio metrics
    avg_snr = np.mean([m['snr_db'] for m in all_metrics])
    avg_corr = np.mean([m['correlation'] for m in all_metrics])
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_l1 = np.mean([m['l1'] for m in all_metrics])
    avg_rms_diff = np.mean([m['rms_diff_db'] for m in all_metrics])

    print(f"\n🎵 AUDIO QUALITY METRICS")
    print(f"   SNR:              {avg_snr:.2f} dB")
    print(f"   Correlation:      {avg_corr:.4f}")
    print(f"   MSE:              {avg_mse:.6f}")
    print(f"   L1:               {avg_l1:.4f}")
    print(f"   RMS Diff:         {avg_rms_diff:.2f} dB")

    # Per-genre breakdown
    print(f"\n🎸 PER-GENRE PERFORMANCE")
    print(f"   {'Genre':<12} {'Avg Loss':>10} {'Samples':>10}")
    print(f"   {'-'*12} {'-'*10} {'-'*10}")

    for genre in GTZAN_GENRES:
        if per_genre_losses[genre]:
            avg = np.mean(per_genre_losses[genre])
            count = len(per_genre_losses[genre])
            print(f"   {genre:<12} {avg:>10.4f} {count:>10}")

    # Best and worst genres
    genre_avgs = {g: np.mean(l) for g, l in per_genre_losses.items() if l}
    best_genre = min(genre_avgs, key=genre_avgs.get)
    worst_genre = max(genre_avgs, key=genre_avgs.get)

    print(f"\n   Best Genre:  {best_genre} (loss: {genre_avgs[best_genre]:.4f})")
    print(f"   Worst Genre: {worst_genre} (loss: {genre_avgs[worst_genre]:.4f})")

    print("\n" + "=" * 70)

    return {
        'total_loss': avg_loss,
        'snr_db': avg_snr,
        'correlation': avg_corr,
        'per_genre': genre_avgs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GenreMaster Model")
    parser.add_argument('--checkpoint', type=str,
                        default='results/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save audio examples')
    parser.add_argument('--num-examples', type=int, default=5,
                        help='Number of audio examples to save')
    parser.add_argument('--output-dir', type=str, default='results/audio_samples',
                        help='Directory to save audio examples')
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("GenreMaster Model Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load dataset
    print("\nLoading GTZAN test set...")
    datasets, genre_to_idx = setup_gtzan(
        audio_dir=Path(config['data']['audio_dir']),
        sr=config['data'].get('sample_rate', 22050),
        duration=config['data'].get('duration', 30.0),
    )

    test_loader = DataLoader(
        datasets['test'],
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"✓ Test samples: {len(datasets['test'])}")

    # Load model
    print("\nLoading model...")
    model = create_genremaster_model(
        n_genres=config['model'].get('n_genres', 10),
        conditioned=config['model'].get('conditioned', True),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Setup loss and transforms
    criterion = create_loss_function(
        sample_rate=config['data'].get('sample_rate', 22050),
        loss_weights=config['loss'].get('weights', None),
    )

    transforms = create_premaster_transforms(
        target_lufs=config['data'].get('target_lufs', -18.0),
        sample_rate=config['data'].get('sample_rate', 22050),
    )

    # Evaluate
    print("\nEvaluating on test set...")
    all_losses, all_metrics, per_genre_losses = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        transforms=transforms,
        device=device,
        save_audio=args.save_audio,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )

    # Print results
    results = print_results(all_losses, all_metrics, per_genre_losses)

    # Save results
    results_path = Path(args.output_dir) / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'test_samples': len(datasets['test']),
            'total_loss': results['total_loss'],
            'snr_db': results['snr_db'],
            'correlation': results['correlation'],
            'per_genre': results['per_genre'],
        }, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    if args.save_audio:
        print(f"✓ Audio examples saved to {args.output_dir}")


if __name__ == '__main__':
    main()
