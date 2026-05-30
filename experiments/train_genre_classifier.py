"""Train genre classifier on FMA Small or GTZAN dataset.

This script trains a simple CNN classifier to predict genre from audio.
Can be used standalone or integrated into GenreMaster.

Usage:
    # Train with GTZAN (10 genres)
    uv run python experiments/train_genre_classifier.py --config configs/gtzan.yaml

    # Train with FMA Small (8 genres)
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
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import setup_fma_medium
from data.gtzan import setup_gtzan, GTZAN_GENRES
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


def load_dataset(config, data_root):
    """Load dataset based on config."""
    dataset_type = config['data'].get('dataset', 'fma')
    audio_dir = Path(config['data']['audio_dir'])

    if dataset_type.lower() == 'gtzan':
        print("\nLoading GTZAN dataset...")
        datasets, genre_to_idx = setup_gtzan(
            audio_dir=audio_dir,
            sr=config['data'].get('sample_rate', 22050),
            duration=config['data'].get('duration', 30.0),
            train_ratio=config['data'].get('train_ratio', 0.8),
            val_ratio=config['data'].get('val_ratio', 0.1),
        )
        genre_names = GTZAN_GENRES
    else:
        print("\nLoading FMA dataset...")
        datasets, genre_to_idx = setup_fma_medium(
            data_root=data_root,
            audio_dir=audio_dir,
            top_k_genres=config['data'].get('top_k_genres', 8),
            samples_per_genre=config['data'].get('samples_per_genre'),
        )
        genre_names = None  # FMA uses genre IDs

    return datasets, genre_to_idx, genre_names


def print_genre_mapping(genre_to_idx, genre_names=None):
    """Print genre mapping."""
    print(f"\n✓ Loaded {len(genre_to_idx)} genres:")
    for genre, idx in sorted(genre_to_idx.items(), key=lambda x: x[1]):
        if genre_names:
            print(f"  {idx}: {genre}")
        else:
            print(f"  {idx}: Genre ID {genre}")


def main():
    parser = argparse.ArgumentParser(description="Train Genre Classifier")
    parser.add_argument('--config', type=str, default='configs/gtzan.yaml',
                        help='Path to config file')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Path to audio directory (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Override config with command line args
    if args.audio_dir:
        config['data']['audio_dir'] = args.audio_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    seed_everything(42)
    device = get_device()

    dataset_type = config['data'].get('dataset', 'fma').upper()
    n_genres = config['data'].get('n_genres', config['model'].get('n_genres', 10))

    print("=" * 70)
    print(f"Genre Classifier Training ({dataset_type} - {n_genres} Genres)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Audio dir: {config['data']['audio_dir']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")

    # Load datasets
    data_root = Path(config['data'].get('root_dir', 'data'))
    datasets, genre_to_idx, genre_names = load_dataset(config, data_root)
    print_genre_mapping(genre_to_idx, genre_names)

    # Create dataloaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device'].get('num_workers', 0),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device'].get('num_workers', 0),
        collate_fn=collate_fn,
    )

    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating CNN classifier...")
    model = GenreCNNClassifier(
        n_genres=len(genre_to_idx),
        n_mels=config['model'].get('n_mels', 128),
        dropout=config['model'].get('dropout', 0.3),
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {n_params:,}")

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    # Get output path from config
    output_dir = Path(config['output'].get('checkpoint_dir', 'results'))
    model_name = config['output'].get('model_name', 'genre_classifier_best.pt')
    save_path = output_dir / model_name

    model = train_genre_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        device=device,
        save_path=str(save_path),
        patience=config['training'].get('patience', 10),
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best model saved to: {save_path}")
    print("\nYou can now use this classifier with GenreMaster:")
    print("  from src.models.genre_classifier import GenreMasterWithCNNClassifier")
    print("  model = GenreMasterWithCNNClassifier(")
    print("      genremaster_model=genremaster,")
    print("      use_pretrained_classifier=True,")
    print(f"      classifier_path='{save_path}'")
    print("  )")


if __name__ == '__main__':
    main()
