"""Tests for core utility functions."""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    get_device,
    seed_everything,
    rms_normalize,
    ensure_stereo,
    ensure_mono,
)


def test_device_detection():
    """Test device detection."""
    device = get_device()
    print(f"✓ Device detected: {device}")
    assert device.type in ["cuda", "mps", "cpu"]


def test_seeding():
    """Test reproducibility with seeding."""
    seed_everything(42)
    x1 = torch.randn(10)
    y1 = np.random.rand(10)

    seed_everything(42)
    x2 = torch.randn(10)
    y2 = np.random.rand(10)

    assert torch.allclose(x1, x2), "PyTorch seeding failed"
    assert np.allclose(y1, y2), "NumPy seeding failed"
    print("✓ Seeding works correctly")


def test_rms_normalize():
    """Test RMS normalization."""
    # Create sine wave
    waveform = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 44100)).unsqueeze(0)
    original_rms = torch.sqrt(torch.mean(waveform ** 2))

    # Normalize to target RMS
    target_rms = 0.1
    normalized = rms_normalize(waveform, target_rms)
    new_rms = torch.sqrt(torch.mean(normalized ** 2))

    print(f"✓ RMS normalize: {original_rms:.4f} → {new_rms:.4f} (target: {target_rms})")
    assert torch.isclose(new_rms, torch.tensor(target_rms), atol=1e-4)


def test_stereo_mono_conversion():
    """Test stereo/mono conversion."""
    # Test mono to stereo
    mono = torch.randn(1, 1000)
    stereo = ensure_stereo(mono)
    assert stereo.shape == (2, 1000), f"Expected (2, 1000), got {stereo.shape}"
    print(f"✓ Mono to stereo: {mono.shape} → {stereo.shape}")

    # Test stereo to mono
    stereo_input = torch.randn(2, 1000)
    mono_output = ensure_mono(stereo_input)
    assert mono_output.shape == (1, 1000), f"Expected (1, 1000), got {mono_output.shape}"
    print(f"✓ Stereo to mono: {stereo_input.shape} → {mono_output.shape}")


if __name__ == "__main__":
    print("Running core utility tests...\n")
    test_device_detection()
    test_seeding()
    test_rms_normalize()
    test_stereo_mono_conversion()
    print("\n✅ All tests passed!")
