"""Generate IEEE paper figures from recorded GenreMaster experiment artifacts.

All figures are created from saved JSON outputs under results/.
No synthetic or placeholder metrics are introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_architecture(output_dir: Path) -> None:
    """Create a professional, large architecture diagram with modern design."""
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stages = [
        "Input\nWaveform",
        "Spectral\nEncoder",
        "Genre\nEmbedding",
        "FiLM\nConditioning",
        "Parameter\nPredictor",
        "Differentiable\nDSP Chain",
        "Enhanced\nOutput",
    ]

    # Color scheme - modern and professional
    colors = ["#e8f4f8", "#d4e6f1", "#b3d9f2", "#85c1e9", "#5dade2", "#3498db", "#2e86c1"]
    edge_color = "#154360"

    # Calculate positions with more spacing
    n_stages = len(stages)
    x_positions = np.linspace(0.08, 0.92, n_stages)
    y = 0.5
    box_width = 0.11
    box_height = 0.35

    for i, (x, text, color) in enumerate(zip(x_positions, stages, colors)):
        # Draw larger rounded boxes
        bbox_props = {
            "boxstyle": "round,pad=0.7",
            "facecolor": color,
            "edgecolor": edge_color,
            "linewidth": 2.5
        }

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=bbox_props,
            transform=ax.transAxes,
        )

        # Draw arrows between stages
        if i < len(stages) - 1:
            arrow_start = x + 0.062
            arrow_end = x_positions[i + 1] - 0.062

            ax.annotate(
                "",
                xy=(arrow_end, y),
                xytext=(arrow_start, y),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": 3.5,
                    "color": edge_color,
                    "shrinkA": 0,
                    "shrinkB": 0
                },
            )

    # Add title with more prominence
    ax.text(
        0.5, 0.92,
        "Genre-Conditioned Audio Enhancement Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Add subtitle explaining the flow
    ax.text(
        0.5, 0.08,
        "Audio features are modulated by genre embeddings via FiLM conditioning to predict DSP parameters",
        ha="center",
        va="center",
        fontsize=11,
        fontstyle="italic",
        color="#566573",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "fig_architecture_pipeline.png", dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def plot_v1_training_curves(output_dir: Path, history: dict) -> None:
    epochs = [e["epoch"] for e in history["epochs"]]
    train_loss = [e["train_loss"] for e in history["epochs"]]
    val_loss = [e["val_loss"] for e in history["epochs"]]

    best_idx = int(np.argmin(val_loss))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_loss, label="Validation loss", linewidth=2.0, color="#d62728")
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], color="#2ca02c", s=60, zorder=5)
    ax.annotate(
        f"Best epoch {epochs[best_idx]}\nVal {val_loss[best_idx]:.4f}",
        xy=(epochs[best_idx], val_loss[best_idx]),
        xytext=(epochs[best_idx] + 1, val_loss[best_idx] + 0.25),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Total loss", fontsize=11)
    ax.set_title("GenreMaster V1 Training and Validation Loss", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v1_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_genre_losses(output_dir: Path, evaluation: dict) -> None:
    per_genre = evaluation["per_genre"]
    genres = list(per_genre.keys())
    losses = [per_genre[g] for g in genres]

    order = np.argsort(losses)
    genres_sorted = [genres[i] for i in order]
    losses_sorted = [losses[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(genres_sorted, losses_sorted, color="#4c78a8", edgecolor="black", linewidth=0.4)

    for bar, value in zip(bars, losses_sorted):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.set_ylabel("Test loss", fontsize=11)
    ax.set_xlabel("Genre", fontsize=11)
    ax.set_title("GenreMaster V1 Test Loss by GTZAN Genre", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v1_per_genre_test_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_v1_v2_comparison(output_dir: Path, comparison: dict) -> None:
    v1 = comparison["v1_baseline"]
    v2 = comparison["v2_transformer"]

    labels = ["V1 (CNN)", "V2 (Transformer)"]
    params_m = [v1["parameters"] / 1e6, v2["parameters"] / 1e6]
    best_val = [v1["best_val_loss"], v2["best_val_loss"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    axes[0].bar(labels, params_m, color=["#2ca02c", "#9467bd"], edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Parameters (millions)", fontsize=11)
    axes[0].set_title("Model Size", fontsize=12)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", labelrotation=15)

    axes[1].bar(labels, best_val, color=["#2ca02c", "#9467bd"], edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Best validation loss", fontsize=11)
    axes[1].set_title("Validation Performance", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", labelrotation=15)

    fig.suptitle("GenreMaster V1 vs V2 (Recorded Comparison Artifacts)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v1_v2_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_comparison(output_dir: Path, cnn_history: dict, resnet_history: dict) -> None:
    """Plot only ResNet classifier performance (CNN excluded due to incomplete training)."""
    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    # Only plot ResNet - CNN training was incomplete
    ax.plot(resnet_history["epoch"], resnet_history["val_acc"], label="ResNet val acc", linewidth=2.0, color="#1f77b4")

    resnet_best = float(max(resnet_history["val_acc"]))
    best_epoch = resnet_history["epoch"][resnet_history["val_acc"].index(resnet_best)]

    # Mark best performance with horizontal line and annotation
    ax.axhline(resnet_best, color="#1f77b4", linestyle="--", alpha=0.5, linewidth=1.0)
    ax.scatter([best_epoch], [resnet_best], color="#2ca02c", s=80, zorder=5, edgecolors='black', linewidths=1.5)

    # Position annotation above and to the right of the best point to avoid overlap
    ax.annotate(
        f"Best: {resnet_best:.2f}% @ epoch {best_epoch}",
        xy=(best_epoch, resnet_best),
        xytext=(best_epoch + 8, resnet_best + 3),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "black"},
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation accuracy (%)", fontsize=11)
    ax.set_title("ResNet Genre Classifier Validation Accuracy (GTZAN)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, loc='lower right')
    fig.tight_layout()
    fig.savefig(output_dir / "fig_classifier_val_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "IEEE_PAPER" / "figures"
    ensure_output_dir(output_dir)

    training_history = load_json(root / "results" / "logs" / "training_history.json")
    evaluation = load_json(root / "results" / "audio_samples" / "evaluation_results.json")
    comparison = load_json(root / "results" / "v2_comparison_results.json")
    cnn_history = load_json(root / "results" / "classifier_history.json")
    resnet_history = load_json(root / "results" / "classifier_resnet_history.json")

    plot_architecture(output_dir)
    plot_v1_training_curves(output_dir, training_history)
    plot_per_genre_losses(output_dir, evaluation)
    plot_v1_v2_comparison(output_dir, comparison)
    plot_classifier_comparison(output_dir, cnn_history, resnet_history)

    print("Generated figures:")
    for p in sorted(output_dir.glob("fig_*.png")):
        print(f" - {p.relative_to(root)}")


if __name__ == "__main__":
    main()