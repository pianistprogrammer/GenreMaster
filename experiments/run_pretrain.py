"""Genre embedding pre-training with contrastive learning.

Pre-trains the genre embedding network using contrastive loss (InfoNCE)
to ensure genre embeddings capture semantic differences.

Usage:
    uv run python experiments/run_pretrain.py --config configs/pretrain.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import yaml
import warnings

# Filter PyTorch STFT resize warnings (harmless deprecation warnings)
warnings.filterwarnings('ignore', message='.*An output with one or more elements was resized.*')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import setup_fma_medium
from models.encoder import create_encoder
from models.genre_embedding import (
    GenreEmbeddingNetwork,
    ContrastiveLoss,
    GenreClassifier,
)
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

    # Limit dataset size
    if config['data'].get('n_train_samples'):
        n_train = min(config['data']['n_train_samples'], len(datasets['train']))
        datasets['train'] = Subset(datasets['train'], range(n_train))

    if config['data'].get('n_val_samples'):
        n_val = min(config['data']['n_val_samples'], len(datasets['val']))
        datasets['val'] = Subset(datasets['val'], range(n_val))

    train_loader = DataLoader(
        datasets['train'],
        batch_size=config['pretraining']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=config['pretraining']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, genre_to_idx


def train_epoch(
    encoder: nn.Module,
    genre_embedding: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    contrastive_loss: nn.Module,
    classification_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    encoder.train()
    genre_embedding.train()
    classifier.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_classification = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        waveform = batch['waveform'].to(device)
        genre_idx = batch['genre_idx'].to(device)

        # Extract audio features
        audio_features = encoder(waveform)

        # Get genre embeddings
        genre_latent = genre_embedding(genre_idx, audio_features)

        # Compute contrastive loss
        loss_contrastive = contrastive_loss(genre_latent, genre_idx)

        # Compute classification loss
        logits = classifier(genre_latent)
        loss_classification = classification_loss(logits, genre_idx)

        # Combined loss
        loss = (
            config['loss']['contrastive_weight'] * loss_contrastive +
            config['loss']['classification_weight'] * loss_classification
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_contrastive += loss_contrastive.item()
        total_classification += loss_classification.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'contrastive': loss_contrastive.item(),
            'cls': loss_classification.item(),
        })

        # Log to Trackio
        if config['logging']['use_trackio'] and batch_idx % config['logging']['log_every'] == 0:
            step = epoch * len(train_loader) + batch_idx
            trackio.log({
                'pretrain/loss': loss.item(),
                'pretrain/contrastive': loss_contrastive.item(),
                'pretrain/classification': loss_classification.item(),
                'pretrain/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'step': step,
            })

    # Average
    avg_loss = total_loss / num_batches
    avg_contrastive = total_contrastive / num_batches
    avg_classification = total_classification / num_batches

    return {
        'total': avg_loss,
        'contrastive': avg_contrastive,
        'classification': avg_classification,
    }


def validate_epoch(
    encoder: nn.Module,
    genre_embedding: nn.Module,
    classifier: nn.Module,
    val_loader: DataLoader,
    contrastive_loss: nn.Module,
    classification_loss: nn.Module,
    device: torch.device,
    config: Dict,
    epoch: int,
) -> Dict[str, float]:
    """Validate for one epoch."""
    encoder.eval()
    genre_embedding.eval()
    classifier.eval()

    total_loss = 0.0
    total_contrastive = 0.0
    total_classification = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            # Extract features
            audio_features = encoder(waveform)
            genre_latent = genre_embedding(genre_idx, audio_features)

            # Losses
            loss_contrastive = contrastive_loss(genre_latent, genre_idx)
            logits = classifier(genre_latent)
            loss_classification = classification_loss(logits, genre_idx)

            loss = (
                config['loss']['contrastive_weight'] * loss_contrastive +
                config['loss']['classification_weight'] * loss_classification
            )

            # Accumulate
            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_classification += loss_classification.item()

            # Accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == genre_idx).sum().item()
            total += genre_idx.size(0)

    # Average
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_contrastive = total_contrastive / num_batches
    avg_classification = total_classification / num_batches
    accuracy = 100.0 * correct / total

    return {
        'total': avg_loss,
        'contrastive': avg_contrastive,
        'classification': avg_classification,
        'accuracy': accuracy,
    }


def save_checkpoint(
    encoder: nn.Module,
    genre_embedding: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    checkpoint_path: Path,
):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'genre_embedding_state_dict': genre_embedding.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train genre embedding")
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    seed_everything(config['experiment']['seed'])

    print("=" * 70)
    print("Genre Embedding Pre-training")
    print("=" * 70)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Config: {args.config}\n")

    # Device
    device = get_device()
    print(f"Device: {device}\n")

    # Create output directories
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['output'].get('log_dir', 'results/logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training history
    training_history = {
        'config': config,
        'experiment_name': config['experiment']['name'],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'epochs': []
    }
    history_path = log_dir / 'pretrain_history.json'

    # Initialize Trackio
    if config['logging']['use_trackio']:
        trackio.init(project=config['logging']['trackio']['project'])
        print("✓ Trackio initialized\n")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, genre_to_idx = create_dataloaders(config)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Genres: {len(genre_to_idx)}\n")

    # Create models
    print("Creating models...")
    encoder = create_encoder(
        encoder_type=config['model']['encoder_type'],
        feature_dim=config['model']['audio_feature_dim'],
        sample_rate=config['data']['sample_rate'],
    ).to(device)

    genre_embedding = GenreEmbeddingNetwork(
        n_genres=len(genre_to_idx),
        embedding_dim=256,
        audio_feature_dim=config['model']['audio_feature_dim'],
        output_dim=config['model']['genre_latent_dim'],
    ).to(device)

    classifier = GenreClassifier(
        input_dim=config['model']['genre_latent_dim'],
        n_genres=len(genre_to_idx),
    ).to(device)

    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in genre_embedding.parameters()) +
        sum(p.numel() for p in classifier.parameters())
    )
    print(f"✓ Total parameters: {total_params:,}\n")

    # Create losses
    contrastive_loss = ContrastiveLoss(temperature=config['pretraining']['temperature'])
    classification_loss = nn.CrossEntropyLoss()

    # Create optimizer
    params = (
        list(encoder.parameters()) +
        list(genre_embedding.parameters()) +
        list(classifier.parameters())
    )

    optimizer = torch.optim.Adam(
        params,
        lr=config['pretraining']['learning_rate'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
        weight_decay=config['pretraining']['weight_decay'],
    )

    # Scheduler
    scheduler = None
    if config['pretraining']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['pretraining']['num_epochs'] - config['pretraining']['warmup_epochs'],
        )

    # Training loop
    print("=" * 70)
    print("Starting Pre-training")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(config['pretraining']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['pretraining']['num_epochs']}")
        epoch_start_time = time.time()

        # Train
        train_losses = train_epoch(
            encoder, genre_embedding, classifier,
            train_loader, contrastive_loss, classification_loss,
            optimizer, device, config, epoch
        )

        print(f"Train loss: {train_losses['total']:.6f} "
              f"(contrastive: {train_losses['contrastive']:.6f}, "
              f"cls: {train_losses['classification']:.6f})")

        # Validate
        val_losses = validate_epoch(
            encoder, genre_embedding, classifier,
            val_loader, contrastive_loss, classification_loss,
            device, config, epoch
        )

        print(f"Val loss: {val_losses['total']:.6f}, "
              f"accuracy: {val_losses['accuracy']:.2f}%")

        epoch_time = time.time() - epoch_start_time

        # Save epoch metrics to history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_losses['total'],
            'train_loss_components': {
                'contrastive': train_losses['contrastive'],
                'classification': train_losses['classification']
            },
            'val_loss': val_losses['total'],
            'val_loss_components': {
                'contrastive': val_losses['contrastive'],
                'classification': val_losses['classification']
            },
            'val_accuracy': val_losses['accuracy'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time_seconds': epoch_time,
            'is_best': val_losses['total'] < best_val_loss
        }
        training_history['epochs'].append(epoch_data)

        # Save history to JSON after each epoch
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        # Log to Trackio
        if config['logging']['use_trackio']:
            trackio.log({
                'val/loss': val_losses['total'],
                'val/contrastive': val_losses['contrastive'],
                'val/classification': val_losses['classification'],
                'val/accuracy': val_losses['accuracy'],
                'epoch': epoch,
            })

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        if (epoch + 1) % config['pretraining']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                encoder, genre_embedding, optimizer,
                epoch, best_val_loss, checkpoint_path
            )

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = checkpoint_dir / "best_pretrained.pt"
            save_checkpoint(
                encoder, genre_embedding, optimizer,
                epoch, best_val_loss, best_path
            )
            print(f"✓ New best model! Val loss: {best_val_loss:.6f}")

    # Save final model (genre embedding only for main training)
    final_path = Path(config['output']['final_model_path'])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'genre_embedding_state_dict': genre_embedding.state_dict(),
        'n_genres': len(genre_to_idx),
        'embedding_dim': 256,
        'output_dim': config['model']['genre_latent_dim'],
    }, final_path)

    # Finalize training history
    training_history['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    training_history['best_val_loss'] = best_val_loss
    training_history['total_epochs'] = len(training_history['epochs'])

    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 70)
    print("Pre-training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Pre-trained genre embedding saved to: {final_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
