"""Test audio feature extraction."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.extractor import AudioFeatureExtractor, extract_features_from_file
from utils import load_audio


def test_feature_extraction():
    """Test feature extraction on a sample audio file."""
    # Use a real FMA track
    audio_path = "/Volumes/LLModels/Datasets/fma_medium/000002.mp3"

    if not Path(audio_path).exists():
        print(f"⚠ Test audio file not found: {audio_path}")
        print("Creating synthetic audio for testing...")

        # Create synthetic stereo audio (5 seconds)
        sr = 44100
        duration = 5.0
        t = torch.linspace(0, duration, int(sr * duration))

        # Left channel: 440 Hz sine wave
        left = torch.sin(2 * torch.pi * 440 * t)

        # Right channel: 880 Hz sine wave (for stereo width)
        right = torch.sin(2 * torch.pi * 880 * t)

        waveform = torch.stack([left, right]) * 0.5  # Scale to prevent clipping

        # Extract features from synthetic audio
        extractor = AudioFeatureExtractor(sr)
        features = extractor.extract_all_features(waveform)

        print("\n" + "=" * 60)
        print("Synthetic Audio Feature Extraction Results")
        print("=" * 60)

    else:
        print("=" * 60)
        print("Audio Feature Extraction Test")
        print("=" * 60)
        print(f"Test file: {audio_path}\n")

        # Load audio
        waveform, sr = load_audio(audio_path, sr=44100, mono=False, duration=30.0)
        print(f"✓ Loaded audio: shape={waveform.shape}, sr={sr}")

        # Initialize extractor
        extractor = AudioFeatureExtractor(sr)

        # Extract individual features
        print("\nExtracting features...")

        lufs = extractor.extract_lufs(waveform)
        print(f"  LUFS (integrated loudness): {lufs:.2f} dB")

        lra = extractor.extract_lra(waveform)
        print(f"  LRA (loudness range): {lra:.2f} LU")

        true_peak = extractor.extract_true_peak(waveform)
        print(f"  True Peak: {true_peak:.2f} dBTP")

        spectral_tilt = extractor.extract_spectral_tilt(waveform)
        print(f"  Spectral Tilt: {spectral_tilt:.2f} dB/octave")

        ms_ratio = extractor.extract_ms_ratio(waveform)
        print(f"  M-S Ratio (stereo width): {ms_ratio:.3f}")

        dr14 = extractor.extract_dr14(waveform)
        print(f"  DR14 (dynamic range): {dr14:.2f} dB")

        # Extract all at once
        features = extractor.extract_all_features(waveform)

        print("\n" + "=" * 60)
        print("All Features (batch extraction):")
        print("=" * 60)

    for feature_name, value in features.items():
        print(f"  {feature_name:20s}: {value:10.3f}")

    # Validate feature ranges
    print("\n" + "=" * 60)
    print("Feature Validation:")
    print("=" * 60)

    validations = [
        ("LUFS", features['lufs'], -70, 0, "dB"),
        ("LRA", features['lra'], 0, 30, "LU"),
        ("True Peak", features['true_peak'], -100, 3, "dBTP"),
        ("Spectral Tilt", features['spectral_tilt'], -20, 20, "dB/oct"),
        ("M-S Ratio", features['ms_ratio'], 0, 2, "ratio"),
        ("DR14", features['dr14'], 0, 30, "dB"),
    ]

    all_valid = True
    for name, value, min_val, max_val, unit in validations:
        if min_val <= value <= max_val:
            status = "✓"
        else:
            status = "✗"
            all_valid = False
        print(f"  {status} {name:20s}: {value:8.2f} {unit:8s} (range: [{min_val}, {max_val}])")

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All features extracted successfully and within expected ranges!")
    else:
        print("⚠ Some features outside expected ranges (may be normal for some audio)")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_extraction()
