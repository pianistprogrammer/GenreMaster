"""Plot training history from JSON logs.

Usage:
    python scripts/plot_training.py results/logs/training_history.json
    python scripts/plot_training.py results/logs/pretrain_history.json
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_main_training(history: dict, output_dir: Path):
    """Plot main training metrics."""
    epochs = [e['epoch'] for e in history['epochs']]
    train_losses = [e['train_loss'] for e in history['epochs']]
    val_losses = [e['val_loss'] for e in history['epochs']]

    # Extract loss components
    train_loudness = [e['train_loss_components']['loudness'] for e in history['epochs']]
    train_spectral = [e['train_loss_components']['spectral'] for e in history['epochs']]
    train_dynamic = [e['train_loss_components']['dynamic'] for e in history['epochs']]
    train_perceptual = [e['train_loss_components']['perceptual'] for e in history['epochs']]

    val_loudness = [e['val_loss_components']['loudness'] for e in history['epochs']]
    val_spectral = [e['val_loss_components']['spectral'] for e in history['epochs']]
    val_dynamic = [e['val_loss_components']['dynamic'] for e in history['epochs']]
    val_perceptual = [e['val_loss_components']['perceptual'] for e in history['epochs']]

    # Learning rate
    learning_rates = [e['learning_rate'] for e in history['epochs']]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training History: {history['experiment_name']}", fontsize=16)

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    ax.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_losses[best_epoch_idx]
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: Epoch {best_epoch}')
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)

    # Plot 2: Loss Components (Training)
    ax = axes[0, 1]
    ax.plot(epochs, train_loudness, label='Loudness', marker='o', markersize=2)
    ax.plot(epochs, train_spectral, label='Spectral', marker='s', markersize=2)
    ax.plot(epochs, train_dynamic, label='Dynamic', marker='^', markersize=2)
    ax.plot(epochs, train_perceptual, label='Perceptual', marker='d', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss Components (Validation)
    ax = axes[1, 0]
    ax.plot(epochs, val_loudness, label='Loudness', marker='o', markersize=2)
    ax.plot(epochs, val_spectral, label='Spectral', marker='s', markersize=2)
    ax.plot(epochs, val_dynamic, label='Dynamic', marker='^', markersize=2)
    ax.plot(epochs, val_perceptual, label='Perceptual', marker='d', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, learning_rates, label='Learning Rate', marker='o', markersize=3, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Experiment: {history['experiment_name']}")
    print(f"Total Epochs: {len(epochs)}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print("=" * 60)


def plot_pretrain(history: dict, output_dir: Path):
    """Plot pre-training metrics."""
    epochs = [e['epoch'] for e in history['epochs']]
    train_losses = [e['train_loss'] for e in history['epochs']]
    val_losses = [e['val_loss'] for e in history['epochs']]
    val_accuracy = [e['val_accuracy'] for e in history['epochs']]

    # Extract loss components
    train_contrastive = [e['train_loss_components']['contrastive'] for e in history['epochs']]
    train_classification = [e['train_loss_components']['classification'] for e in history['epochs']]

    val_contrastive = [e['val_loss_components']['contrastive'] for e in history['epochs']]
    val_classification = [e['val_loss_components']['classification'] for e in history['epochs']]

    # Learning rate
    learning_rates = [e['learning_rate'] for e in history['epochs']]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Pre-training History: {history['experiment_name']}", fontsize=16)

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    ax.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_losses[best_epoch_idx]
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)

    # Plot 2: Loss Components
    ax = axes[0, 1]
    ax.plot(epochs, train_contrastive, label='Train Contrastive', marker='o', markersize=2)
    ax.plot(epochs, train_classification, label='Train Classification', marker='s', markersize=2)
    ax.plot(epochs, val_contrastive, label='Val Contrastive', marker='^', markersize=2)
    ax.plot(epochs, val_classification, label='Val Classification', marker='d', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, val_accuracy, label='Val Accuracy', marker='o', markersize=3, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, learning_rates, label='Learning Rate', marker='o', markersize=3, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'pretrain_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Pre-training Summary")
    print("=" * 60)
    print(f"Experiment: {history['experiment_name']}")
    print(f"Total Epochs: {len(epochs)}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final Val Accuracy: {val_accuracy[-1]:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument('history_file', type=str, help='Path to training_history.json or pretrain_history.json')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    # Load history
    history_path = Path(args.history_file)
    if not history_path.exists():
        print(f"Error: {history_path} not found")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect type and plot
    if 'pretrain' in history_path.name:
        plot_pretrain(history, output_dir)
    else:
        plot_main_training(history, output_dir)


if __name__ == "__main__":
    main()
