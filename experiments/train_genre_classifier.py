"""Train genre classifier on FMA Small (8 balanced genres).

This script trains a simple CNN classifier to predict genre from audio.
Can be used standalone or integrated into GenreMaster.

Usage:
    # Train classifier only
    uv run python experiments/train_genre_classifier.py --config configs/classifier.yaml

    # Then use with GenreMaster
    model = GenreMasterWithCNNClassifier(
        genremaster_model=genremaster,
        use_pretrained_classifier=True,
        classifier_path='results/genre_classifier_best.pt'
    )
"""

import argparse
import sys
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import setup_fma_medium  # Works for both small and medium
from models.genre_classifier import GenreCNNClassifier, train_genre_classifier
from utils import seed_everything, get_device


def collate_fn(batch):
    """Custom collate function."""
    min_length = min(item['waveform'].shape[1] for item in batch)
    waveforms = torch.stack([item['waveform'][:, :min_length] for item in batch])
    genre_indices = torch.tensor([item['genre_idx'] for item in batch])

    return {
        'waveform': waveforms,
        'genre_idx': genre_indices,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Genre Classifier")
    parser.add_argument('--config', type=str, default='configs/classifier.yaml',
                        help='Path to config file')
    parser.add_argument('--audio_dir', type=str,
                        default='/Volumes/LLModels/Datasets/fma_small',
                        help='Path to FMA audio directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    # Load config if exists
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'data': {
                'audio_dir': args.audio_dir,
                'top_k_genres': 8,
                'samples_per_genre': None,  # Use all
            },
            'training': {
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'learning_rate': args.lr,
            },
            'device': {'type': 'auto'},
        }

    seed_everything(42)
    device = get_device()

    print("=" * 70)
    print("Genre Classifier Training (FMA Small - 8 Genres)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Audio dir: {config['data']['audio_dir']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")

    # Load datasets
    print("\nLoading FMA Small dataset...")
    datasets, genre_to_idx = setup_fma_medium(
        data_root=Path('data'),
        audio_dir=Path(config['data']['audio_dir']),
        top_k_genres=config['data']['top_k_genres'],
        samples_per_genre=config['data'].get('samples_per_genre'),
    )

    print(f"\n✓ Loaded {len(genre_to_idx)} genres:")
    for genre_id, idx in sorted(genre_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: Genre {genre_id}")

    # Create dataloaders
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

    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating CNN classifier...")
    model = GenreCNNClassifier(n_genres=len(genre_to_idx))

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {n_params:,}")

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    model = train_genre_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        device=device,
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best model saved to: results/genre_classifier_best.pt")
    print("\nYou can now use this classifier with GenreMaster:")
    print("  from src.models.genre_classifier import GenreMasterWithCNNClassifier")
    print("  model = GenreMasterWithCNNClassifier(")
    print("      genremaster_model=genremaster,")
    print("      use_pretrained_classifier=True,")
    print("      classifier_path='results/genre_classifier_best.pt'")
    print("  )")


if __name__ == '__main__':
    main()
