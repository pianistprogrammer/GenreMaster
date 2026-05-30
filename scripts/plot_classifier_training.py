#!/usr/bin/env python
"""Plot training curves for genre classifier.

Usage:
    uv run python scripts/plot_classifier_training.py
    uv run python scripts/plot_classifier_training.py --history results/classifier_history.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(history_path: str) -> dict:
    """Load training history from JSON file."""
    with open(history_path) as f:
        return json.load(f)


def plot_loss_curves(history: dict, save_path: str = None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history['epoch']
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Genre Classifier - Loss Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved loss plot to: {save_path}")
    plt.show()


def plot_accuracy_curves(history: dict, save_path: str = None):
    """Plot training and validation accuracy curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history['epoch']
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)

    # Mark best validation accuracy
    best_idx = history['val_acc'].index(max(history['val_acc']))
    best_epoch = epochs[best_idx]
    best_acc = history['val_acc'][best_idx]
    ax.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5, label=f'Best Val: {best_acc:.2f}%')
    ax.scatter([best_epoch], [best_acc], color='g', s=100, zorder=5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Genre Classifier - Accuracy Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved accuracy plot to: {save_path}")
    plt.show()


def plot_lr_schedule(history: dict, save_path: str = None):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = history['epoch']
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved LR plot to: {save_path}")
    plt.show()


def plot_combined(history: dict, save_path: str = None):
    """Plot combined loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history['epoch']

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)

    # Mark best
    best_idx = history['val_acc'].index(max(history['val_acc']))
    best_acc = history['val_acc'][best_idx]
    axes[1].axhline(y=best_acc, color='g', linestyle='--', alpha=0.5)
    axes[1].scatter([epochs[best_idx]], [best_acc], color='g', s=100, zorder=5,
                    label=f'Best: {best_acc:.2f}%')

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Genre Classifier Training', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved combined plot to: {save_path}")
    plt.show()


def print_summary(history: dict):
    """Print training summary."""
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)

    n_epochs = len(history['epoch'])
    best_val_acc = history.get('best_val_acc', max(history['val_acc']))
    best_idx = history['val_acc'].index(max(history['val_acc']))
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]

    print(f"Epochs trained: {n_epochs}")
    print(f"Best val accuracy: {best_val_acc:.2f}% (epoch {history['epoch'][best_idx]})")
    print(f"Final train accuracy: {final_train_acc:.2f}%")
    print(f"Final val accuracy: {final_val_acc:.2f}%")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")

    if 'config' in history:
        print(f"\nConfig:")
        for k, v in history['config'].items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Plot classifier training curves")
    parser.add_argument('--history', type=str, default='results/classifier_history.json',
                        help='Path to training history JSON')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Directory to save plots')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots to files')
    args = parser.parse_args()

    # Load history
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        print("Run training first to generate the history file.")
        return

    history = load_history(history_path)
    print(f"✓ Loaded history from: {history_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print_summary(history)

    # Plot
    save_combined = None if args.no_save else str(output_dir / "classifier_training.png")
    plot_combined(history, save_path=save_combined)


if __name__ == '__main__':
    main()
