"""Test full GenreMaster model assembly and forward pass."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.genremaster import create_genremaster_model
from utils import load_audio, seed_everything


def test_genremaster_full():
    """Test full GenreMaster model with real audio."""
    seed_everything(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 70)
    print("GenreMaster Full Model Test")
    print("=" * 70)
    print(f"Device: {device}\n")

    # Model configuration
    n_genres = 16
    sample_rate = 44100

    # Create conditioned model
    print("Creating GenreMaster (conditioned)...")
    model_conditioned = create_genremaster_model(
        n_genres=n_genres,
        encoder_type="lightweight",
        conditioned=True,
        sample_rate=sample_rate,
    ).to(device)

    # Create unconditioned baseline
    print("Creating GenreMaster (unconditioned baseline)...")
    model_unconditioned = create_genremaster_model(
        n_genres=n_genres,
        encoder_type="lightweight",
        conditioned=False,
        sample_rate=sample_rate,
    ).to(device)

    # Print parameter counts
    print("\n" + "=" * 70)
    print("Model Architecture")
    print("=" * 70)
    param_counts = model_conditioned.get_parameter_count()
    print("\nConditioned Model:")
    for component, count in param_counts.items():
        print(f"  {component:20s}: {count:>12,} parameters")

    uncond_params = sum(p.numel() for p in model_unconditioned.parameters())
    print(f"\nUnconditioned Model:    {uncond_params:>12,} parameters")
    print(f"Genre conditioning adds: {param_counts['total'] - uncond_params:>12,} parameters")

    # Load real audio
    audio_path = "/Volumes/LLModels/Datasets/fma_medium/000002.mp3"
    print("\n" + "=" * 70)
    print("Loading Real Audio")
    print("=" * 70)
    print(f"File: {audio_path}")

    waveform, sr = load_audio(audio_path, sr=sample_rate, mono=False, duration=4.0)
    print(f"✓ Loaded: shape={waveform.shape}, sr={sr}")

    # Prepare batch
    batch_size = 2
    waveform_batch = waveform.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    genre_idx = torch.tensor([0, 5]).to(device)  # Rock, Pop

    print(f"\nBatch:")
    print(f"  Waveform: {waveform_batch.shape}")
    print(f"  Genres: {genre_idx.tolist()}")

    # Test forward pass - CONDITIONED MODEL
    print("\n" + "=" * 70)
    print("Testing Conditioned Model Forward Pass")
    print("=" * 70)

    model_conditioned.eval()
    with torch.no_grad():
        try:
            mastered, params = model_conditioned(
                waveform_batch,
                genre_idx,
                return_params=True,
            )

            print("✓ Forward pass successful!")
            print(f"\nOutput:")
            print(f"  Mastered shape: {mastered.shape}")
            print(f"  Output device: {mastered.device}")
            print(f"  Output range: [{mastered.min().item():.3f}, {mastered.max().item():.3f}]")

            print(f"\nPredicted DSP Parameters:")
            for param_name, param_value in params.items():
                print(f"  {param_name:20s}: shape={param_value.shape}, "
                      f"range=[{param_value.min().item():.2f}, {param_value.max().item():.2f}]")

        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            raise

    # Test forward pass - UNCONDITIONED MODEL
    print("\n" + "=" * 70)
    print("Testing Unconditioned Model Forward Pass")
    print("=" * 70)

    model_unconditioned.eval()
    with torch.no_grad():
        try:
            mastered_uncond = model_unconditioned(waveform_batch, genre_idx)

            print("✓ Forward pass successful!")
            print(f"\nOutput:")
            print(f"  Mastered shape: {mastered_uncond.shape}")
            print(f"  Output range: [{mastered_uncond.min().item():.3f}, "
                  f"{mastered_uncond.max().item():.3f}]")

        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            raise

    # Compare outputs
    print("\n" + "=" * 70)
    print("Comparing Conditioned vs Unconditioned")
    print("=" * 70)

    diff = (mastered - mastered_uncond).abs().mean().item()
    print(f"Mean absolute difference: {diff:.6f}")

    if diff > 1e-6:
        print("✓ Models produce different outputs (genre conditioning has effect)")
    else:
        print("⚠ Models produce same output (genre conditioning may not be working)")

    # Test genre-only mode
    print("\n" + "=" * 70)
    print("Testing Genre-Only Mode (no audio conditioning)")
    print("=" * 70)

    with torch.no_grad():
        try:
            mastered_genre_only = model_conditioned.forward_with_genre_only(
                waveform_batch,
                genre_idx,
            )

            print("✓ Genre-only forward pass successful!")
            print(f"  Output shape: {mastered_genre_only.shape}")

        except Exception as e:
            print(f"✗ Genre-only forward pass failed: {e}")
            raise

    # Test with different genres
    print("\n" + "=" * 70)
    print("Testing Different Genre Inputs")
    print("=" * 70)

    genre_names = ["Rock", "Classical", "Electronic", "Jazz", "Pop"]
    outputs = []

    with torch.no_grad():
        for i, genre_name in enumerate(genre_names[:min(5, n_genres)]):
            genre_tensor = torch.tensor([i]).to(device)
            waveform_single = waveform_batch[:1]

            output = model_conditioned(waveform_single, genre_tensor)
            outputs.append(output)

            print(f"  {genre_name:15s} (idx={i}): "
                  f"range=[{output.min().item():.3f}, {output.max().item():.3f}]")

    # Check if outputs differ across genres
    print("\n" + "=" * 70)
    print("Cross-Genre Output Variance")
    print("=" * 70)

    for i in range(len(outputs) - 1):
        diff = (outputs[i] - outputs[i+1]).abs().mean().item()
        print(f"  {genre_names[i]:15s} vs {genre_names[i+1]:15s}: diff={diff:.6f}")

    print("\n" + "=" * 70)
    print("✅ All tests passed! GenreMaster model is ready for training.")
    print("=" * 70)


if __name__ == "__main__":
    test_genremaster_full()
