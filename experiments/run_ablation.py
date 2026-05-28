"""Ablation study: compare conditioned vs unconditioned variants.

Runs multiple model variants to measure the contribution of genre conditioning.

Usage:
    uv run python experiments/run_ablation.py --config configs/ablation.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import yaml
import json
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import setup_fma_medium
from data.transforms import create_premaster_transforms
from models.genremaster import create_genremaster_model
from losses import create_loss_function
from utils import seed_everything, get_device
import trackio


def collate_fn(batch):
    """Custom collate function."""
    min_length = min(item['waveform'].shape[1] for item in batch)
    waveforms = torch.stack([item['waveform'][:, :min_length] for item in batch])
    genre_indices = torch.tensor([item['genre_idx'] for item in batch])
    track_ids = [item['track_id'] for item in batch]

    return {
        'waveform': waveforms,
        'genre_idx': genre_indices,
        'track_id': track_ids,
    }


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict):
    """Create dataloaders."""
    datasets, genre_to_idx = setup_fma_medium(
        data_root=Path(config['data']['root_dir']),
        audio_dir=Path(config['data']['audio_dir']),
        top_k_genres=config['data']['top_k_genres'],
        samples_per_genre=config['data']['samples_per_genre'],
    )

    # Limit dataset
    if config['data'].get('n_train_samples'):
        n_train = min(config['data']['n_train_samples'], len(datasets['train']))
        datasets['train'] = Subset(datasets['train'], range(n_train))

    if config['data'].get('n_val_samples'):
        n_val = min(config['data']['n_val_samples'], len(datasets['val']))
        datasets['val'] = Subset(datasets['val'], range(n_val))

    train_loader = DataLoader(
        datasets['train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, genre_to_idx


def train_model_variant(
    variant_config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    genre_to_idx: Dict,
    transforms: tuple,
    criterion: nn.Module,
    device: torch.device,
    global_config: Dict,
) -> Dict:
    """Train a single model variant."""
    variant_name = variant_config['name']
    model_config = variant_config['model']

    print(f"\n{'=' * 70}")
    print(f"Training Variant: {variant_name}")
    print(f"{'=' * 70}")
    print(f"Conditioned: {model_config.get('conditioned', True)}")
    print(f"Encoder: {model_config.get('encoder_type', 'lightweight')}\n")

    # Create model
    model = create_genremaster_model(
        n_genres=len(genre_to_idx),
        encoder_type=model_config.get('encoder_type', 'lightweight'),
        conditioned=model_config.get('conditioned', True),
        sample_rate=global_config['data']['sample_rate'],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=global_config['training']['learning_rate'],
        betas=global_config['optimizer']['betas'],
        weight_decay=global_config['training']['weight_decay'],
    )

    # Scheduler
    scheduler = None
    if global_config['training']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=global_config['training']['num_epochs'] - global_config['training']['warmup_epochs'],
        )

    train_transform, val_transform = transforms

    # Training loop
    best_val_loss = float('inf')
    train_losses_history = []
    val_losses_history = []

    for epoch in range(global_config['training']['num_epochs']):
        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            # Create pre-master and targets
            pre_masters = []
            targets = []
            for i in range(waveform.shape[0]):
                pre_master, target = train_transform(waveform[i])
                pre_masters.append(pre_master)
                targets.append(target)

            pre_master_batch = torch.stack(pre_masters).to(device)
            target_batch = torch.stack(targets).to(device)

            # Forward
            optimizer.zero_grad()
            output = model(pre_master_batch, genre_idx)
            loss = criterion(output, target_batch)

            # Backward
            loss.backward()

            if global_config['training'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    global_config['training']['grad_clip']
                )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / num_batches
        train_losses_history.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                waveform = batch['waveform'].to(device)
                genre_idx = batch['genre_idx'].to(device)

                pre_masters = []
                targets = []
                for i in range(waveform.shape[0]):
                    pre_master, target = val_transform(waveform[i])
                    pre_masters.append(pre_master)
                    targets.append(target)

                pre_master_batch = torch.stack(pre_masters).to(device)
                target_batch = torch.stack(targets).to(device)

                output = model(pre_master_batch, genre_idx)
                loss = criterion(output, target_batch)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_losses_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if scheduler is not None:
            scheduler.step()

    # Return results
    return {
        'variant_name': variant_name,
        'conditioned': model_config.get('conditioned', True),
        'encoder_type': model_config.get('encoder_type', 'lightweight'),
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses_history[-1],
        'final_val_loss': val_losses_history[-1],
        'train_losses': train_losses_history,
        'val_losses': val_losses_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--config', type=str, default='configs/ablation.yaml')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    seed_everything(config['experiment']['seed'])

    print("=" * 70)
    print("GenreMaster Ablation Study")
    print("=" * 70)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Variants: {len(config['ablation']['variants'])}\n")

    # Device
    device = get_device()
    print(f"Device: {device}\n")

    # Create output directories
    output_dir = Path(config['output']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Trackio
    if config['logging']['use_trackio']:
        trackio.init(project=config['logging']['trackio']['project'])
        print("✓ Trackio initialized\n")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, genre_to_idx = create_dataloaders(config)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")

    # Create transforms
    transforms = create_premaster_transforms(
        target_lufs=config['data']['target_lufs'],
        sample_rate=config['data']['sample_rate'],
        augment_train=config['data']['augment_train'],
    )

    # Create loss
    criterion = create_loss_function(
        sample_rate=config['data']['sample_rate'],
        loss_weights=config['loss']['weights'],
    )

    # Run each variant
    results = []

    for variant_config in config['ablation']['variants']:
        result = train_model_variant(
            variant_config,
            train_loader,
            val_loader,
            genre_to_idx,
            transforms,
            criterion,
            device,
            config,
        )
        results.append(result)

    # Save comparison results
    print("\n" + "=" * 70)
    print("Ablation Study Results")
    print("=" * 70)

    comparison_df = pd.DataFrame([{
        'Variant': r['variant_name'],
        'Conditioned': r['conditioned'],
        'Encoder': r['encoder_type'],
        'Parameters': r['param_count'],
        'Best Val Loss': r['best_val_loss'],
        'Final Val Loss': r['final_val_loss'],
    } for r in results])

    print(comparison_df.to_string(index=False))

    # Save to CSV
    comparison_file = Path(config['output']['comparison_file'])
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison saved to: {comparison_file}")

    # Save detailed results
    detailed_file = output_dir / "detailed_results.json"
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {detailed_file}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    # Find best variant
    best_variant = min(results, key=lambda x: x['best_val_loss'])
    print(f"Best variant: {best_variant['variant_name']}")
    print(f"  Best val loss: {best_variant['best_val_loss']:.6f}")

    # Compare conditioned vs unconditioned
    conditioned_results = [r for r in results if r['conditioned']]
    unconditioned_results = [r for r in results if not r['conditioned']]

    if conditioned_results and unconditioned_results:
        avg_conditioned = sum(r['best_val_loss'] for r in conditioned_results) / len(conditioned_results)
        avg_unconditioned = sum(r['best_val_loss'] for r in unconditioned_results) / len(unconditioned_results)

        improvement = ((avg_unconditioned - avg_conditioned) / avg_unconditioned) * 100

        print(f"\nConditioned vs Unconditioned:")
        print(f"  Avg conditioned loss: {avg_conditioned:.6f}")
        print(f"  Avg unconditioned loss: {avg_unconditioned:.6f}")
        print(f"  Improvement: {improvement:.2f}%")

    print("\n" + "=" * 70)
    print("Ablation Study Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
