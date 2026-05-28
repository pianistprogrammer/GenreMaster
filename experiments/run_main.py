"""Main training script for GenreMaster with Trackio logging.

Usage:
    uv run python experiments/run_main.py --config configs/default.yaml
    uv run python experiments/run_main.py --config configs/default.yaml --resume results/checkpoints/latest.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
import json
import time

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
from utils import seed_everything, get_device, save_audio
import trackio


def collate_fn(batch):
    """Custom collate function for variable-length audio."""
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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict):
    """Create train and validation dataloaders."""
    # Setup dataset
    datasets, genre_to_idx = setup_fma_medium(
        data_root=Path(config['data']['root_dir']),
        audio_dir=Path(config['data']['audio_dir']),
        top_k_genres=config['data']['top_k_genres'],
        samples_per_genre=config['data']['samples_per_genre'],
    )

    # Limit dataset size if specified
    if config['data'].get('n_train_samples'):
        n_train = min(config['data']['n_train_samples'], len(datasets['train']))
        datasets['train'] = Subset(datasets['train'], range(n_train))

    if config['data'].get('n_val_samples'):
        n_val = min(config['data']['n_val_samples'], len(datasets['val']))
        datasets['val'] = Subset(datasets['val'], range(n_val))

    # Create dataloaders
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_loss: float,
    config: Dict,
    checkpoint_path: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    print(f"  Resuming from epoch {epoch}, best val loss: {best_val_loss:.6f}")

    return epoch, best_val_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    transforms: tuple,
    config: Dict,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    train_transform, _ = transforms

    total_loss = 0.0
    loss_components = {'loudness': 0.0, 'spectral': 0.0, 'dynamic': 0.0, 'perceptual': 0.0}
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
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

        # Forward pass
        optimizer.zero_grad()
        output = model(pre_master_batch, genre_idx)

        # Compute loss
        losses = criterion(output, target_batch, return_components=True)

        # Backward pass
        losses['total'].backward()

        # Gradient clipping
        if config['training'].get('grad_clip'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip']
            )

        optimizer.step()

        # Accumulate losses
        total_loss += losses['total'].item()
        for key in loss_components:
            loss_components[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': losses['total'].item()})

        # Log to Trackio
        if config['logging']['use_trackio'] and batch_idx % config['logging']['log_every'] == 0:
            step = epoch * len(train_loader) + batch_idx
            trackio.log({
                'train/loss': losses['total'].item(),
                'train/loss_loudness': losses['loudness'].item(),
                'train/loss_spectral': losses['spectral'].item(),
                'train/loss_dynamic': losses['dynamic'].item(),
                'train/loss_perceptual': losses['perceptual'].item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'step': step,
            })

    # Average losses
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'total': avg_loss, **loss_components}


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    transforms: tuple,
    epoch: int,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    _, val_transform = transforms

    total_loss = 0.0
    loss_components = {'loudness': 0.0, 'spectral': 0.0, 'dynamic': 0.0, 'perceptual': 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation"):
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            # Create pre-master and targets
            pre_masters = []
            targets = []
            for i in range(waveform.shape[0]):
                pre_master, target = val_transform(waveform[i])
                pre_masters.append(pre_master)
                targets.append(target)

            pre_master_batch = torch.stack(pre_masters).to(device)
            target_batch = torch.stack(targets).to(device)

            # Forward pass
            output = model(pre_master_batch, genre_idx)

            # Compute loss
            losses = criterion(output, target_batch, return_components=True)

            # Accumulate
            total_loss += losses['total'].item()
            for key in loss_components:
                loss_components[key] += losses[key].item()
            num_batches += 1

    # Average
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'total': avg_loss, **loss_components}


def main():
    parser = argparse.ArgumentParser(description="Train GenreMaster model")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    seed_everything(config['experiment']['seed'])

    print("=" * 70)
    print("GenreMaster Training")
    print("=" * 70)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Config: {args.config}\n")

    # Setup device
    device = get_device()
    print(f"Device: {device}\n")

    # Create output directories
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Trackio
    if config['logging']['use_trackio']:
        trackio.init(
            project=config['logging']['trackio']['project'],
        )
        print("✓ Trackio initialized\n")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, genre_to_idx = create_dataloaders(config)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Genres: {len(genre_to_idx)}\n")

    # Create transforms
    train_transform, val_transform = create_premaster_transforms(
        target_lufs=config['data']['target_lufs'],
        sample_rate=config['data']['sample_rate'],
        augment_train=config['data']['augment_train'],
    )

    # Create model
    print("Creating model...")
    model = create_genremaster_model(
        n_genres=len(genre_to_idx),
        encoder_type=config['model']['encoder_type'],
        conditioned=config['model']['conditioned'],
        sample_rate=config['data']['sample_rate'],
    ).to(device)

    param_counts = model.get_parameter_count()
    print(f"✓ Total parameters: {param_counts['total']:,}\n")

    # Create loss
    criterion = create_loss_function(
        sample_rate=config['data']['sample_rate'],
        loss_weights=config['loss']['weights'],
    )

    # Create optimizer
    if config['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
            weight_decay=config['training']['weight_decay'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']['name']}")

    # Create scheduler
    scheduler = None
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'] - config['training']['warmup_epochs'],
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch

    # Training loop
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            (train_transform, val_transform), config, epoch
        )

        print(f"Train loss: {train_losses['total']:.6f}")

        # Validate
        val_losses = validate_epoch(
            model, val_loader, criterion, device,
            (train_transform, val_transform), epoch
        )

        print(f"Val loss: {val_losses['total']:.6f}")

        # Log to Trackio
        if config['logging']['use_trackio']:
            trackio.log({
                'val/loss': val_losses['total'],
                'val/loss_loudness': val_losses['loudness'],
                'val/loss_spectral': val_losses['spectral'],
                'val/loss_dynamic': val_losses['dynamic'],
                'val/loss_perceptual': val_losses['perceptual'],
                'epoch': epoch,
            })

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                best_val_loss, config, checkpoint_path
            )

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                best_val_loss, config, best_path
            )
            print(f"✓ New best model saved! Val loss: {best_val_loss:.6f}")

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(
        model, optimizer, scheduler,
        config['training']['num_epochs'] - 1,
        best_val_loss, config, final_path
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
