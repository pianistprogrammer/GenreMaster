#!/usr/bin/env python
"""Train GenreMaster V2 model with Transformer encoder.

Usage:
    uv run python experiments/train_v2.py
    uv run python experiments/train_v2.py --config configs/default_v2.yaml
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*stft.*')
warnings.filterwarnings('ignore', message='.*Clipping.*')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.gtzan import setup_gtzan
from data.transforms import create_premaster_transforms
from losses import create_loss_function
from models.genremaster_v2 import create_genremaster_v2
from utils import seed_everything, get_device

try:
    import trackio
    HAS_TRACKIO = True
except ImportError:
    HAS_TRACKIO = False


def collate_fn(batch):
    """Collate function for dataloader."""
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
    }


def train_epoch(model, loader, criterion, optimizer, device, transforms, config, epoch):
    """Train for one epoch."""
    model.train()
    train_tf, _ = transforms

    total_loss = 0.0
    loss_components = {'loudness': 0, 'spectral': 0, 'dynamic': 0, 'perceptual': 0}
    num_batches = 0
    skipped_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        waveform = batch['waveform'].to(device)
        genre_idx = batch['genre_idx'].to(device)

        # Create pre-master and target pairs
        pre_masters, targets = [], []
        for i in range(waveform.shape[0]):
            pm, tgt = train_tf(waveform[i])
            pre_masters.append(pm)
            targets.append(tgt)

        pre_master_batch = torch.stack(pre_masters).to(device)
        target_batch = torch.stack(targets).to(device)

        # Forward pass
        try:
            output = model(pre_master_batch, genre_idx)
        except Exception as e:
            print(f"\n[WARNING] Forward pass error at batch {batch_idx}: {e}")
            skipped_batches += 1
            continue

        # Check for NaN in output
        if torch.isnan(output).any():
            print(f"\n[WARNING] NaN in output at batch {batch_idx}, skipping")
            skipped_batches += 1
            continue

        # Compute loss
        losses = criterion(output, target_batch, return_components=True)

        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            print(f"\n[WARNING] NaN/Inf loss at batch {batch_idx}, skipping")
            skipped_batches += 1
            continue

        # Backward
        losses['total'].backward()

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan = True
                break

        if has_nan:
            print(f"\n[WARNING] NaN gradient at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            skipped_batches += 1
            continue

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training'].get('grad_clip', 1.0)
        )

        optimizer.step()

        # Accumulate losses
        total_loss += losses['total'].item()
        for key in loss_components:
            loss_components[key] += losses[key].item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{losses['total'].item():.3f}"})

    if num_batches == 0:
        return {'total': float('inf'), **{k: 0 for k in loss_components}}

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    if skipped_batches > 0:
        print(f"  (Skipped {skipped_batches} batches due to NaN)")

    return {'total': avg_loss, **loss_components}


def validate_epoch(model, loader, criterion, device, transforms):
    """Validate for one epoch."""
    model.eval()
    _, val_tf = transforms

    total_loss = 0.0
    loss_components = {'loudness': 0, 'spectral': 0, 'dynamic': 0, 'perceptual': 0}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            pre_masters, targets = [], []
            for i in range(waveform.shape[0]):
                pm, tgt = val_tf(waveform[i])
                pre_masters.append(pm)
                targets.append(tgt)

            pre_master_batch = torch.stack(pre_masters).to(device)
            target_batch = torch.stack(targets).to(device)

            output = model(pre_master_batch, genre_idx)
            losses = criterion(output, target_batch, return_components=True)

            if not torch.isnan(losses['total']):
                total_loss += losses['total'].item()
                for key in loss_components:
                    loss_components[key] += losses[key].item()
                num_batches += 1

    if num_batches == 0:
        return {'total': float('inf'), **{k: 0 for k in loss_components}}

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'total': avg_loss, **loss_components}


def main():
    parser = argparse.ArgumentParser(description="Train GenreMaster V2")
    parser.add_argument('--config', type=str, default='configs/default_v2.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(config['experiment'].get('seed', 42))
    device = get_device()

    print("=" * 70)
    print("GenreMaster V2 Training")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    # Initialize Trackio
    if HAS_TRACKIO and config['logging'].get('use_trackio', False):
        trackio.init(
            project=config['logging']['trackio']['project'],
        )
        print("✓ Trackio initialized")

    # Load dataset
    print("\nLoading datasets...")
    datasets, genre_to_idx = setup_gtzan(
        audio_dir=Path(config['data']['audio_dir']),
        sr=config['data'].get('sample_rate', 22050),
        duration=config['data'].get('duration', 30.0),
    )

    train_loader = DataLoader(
        datasets['train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_genremaster_v2(
        n_genres=config['model'].get('n_genres', 10),
        sample_rate=config['data'].get('sample_rate', 22050),
        use_residual=config['model'].get('use_residual', True),
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4),
    )

    # Learning rate scheduler
    total_steps = config['training']['num_epochs'] * len(train_loader)
    warmup_steps = config['training'].get('warmup_epochs', 5) * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = create_loss_function(
        sample_rate=config['data'].get('sample_rate', 22050),
        loss_weights=config['loss'].get('weights', None),
    )

    # Transforms
    transforms = create_premaster_transforms(
        target_lufs=config['data'].get('target_lufs', -18.0),
        sample_rate=config['data'].get('sample_rate', 22050),
    )

    # Create output directories
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'config': config,
        'experiment_name': config['experiment']['name'],
        'start_time': datetime.now().isoformat(),
        'device': str(device),
        'epochs': [],
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            transforms, config, epoch + 1
        )
        print(f"Train loss: {train_losses['total']:.4f}")

        # Update scheduler
        scheduler.step()

        # Validate
        val_losses = validate_epoch(
            model, val_loader, criterion, device, transforms
        )
        print(f"Val loss: {val_losses['total']:.4f}")

        # Record history
        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'lr': scheduler.get_last_lr()[0],
        })

        # Log to Trackio
        if HAS_TRACKIO and config['logging'].get('use_trackio', False):
            trackio.log({
                'v2/train_loss': train_losses['total'],
                'v2/val_loss': val_losses['total'],
                'v2/lr': scheduler.get_last_lr()[0],
            })

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model_v2.pt")
            print(f"✓ New best model! Val loss: {val_losses['total']:.4f}")
        else:
            epochs_without_improvement += 1

        # Save checkpoint
        if (epoch + 1) % config['training'].get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
            }, checkpoint_dir / f"checkpoint_v2_epoch_{epoch + 1}.pt")

        # Early stopping
        if epochs_without_improvement >= config['training'].get('patience', 15):
            print(f"\nEarly stopping: no improvement for {epochs_without_improvement} epochs")
            break

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model_v2.pt")

    # Save history
    history['end_time'] = datetime.now().isoformat()
    history['best_val_loss'] = best_val_loss
    history['total_epochs'] = len(history['epochs'])

    with open(log_dir / "training_history_v2.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model_v2.pt'}")
    print(f"History saved to: {log_dir / 'training_history_v2.json'}")


if __name__ == '__main__':
    main()
