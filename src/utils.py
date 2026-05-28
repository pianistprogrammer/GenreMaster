"""Core utility functions for GenreMaster project."""

import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf


def get_device() -> torch.device:
    """
    Detect and return the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device: The detected device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audio(
    path: Union[str, Path],
    sr: int = 44100,
    mono: bool = False,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """
    Load audio file and return as PyTorch tensor.

    Args:
        path: Path to audio file
        sr: Target sample rate (default: 44100)
        mono: Convert to mono if True (default: False)
        duration: Duration to load in seconds (None = full file)
        offset: Start time offset in seconds (default: 0.0)

    Returns:
        Tuple of (audio tensor of shape [channels, samples], sample_rate)
    """
    path = Path(path)

    try:
        # Try torchaudio first
        waveform, sample_rate = torchaudio.load(str(path))

        # Resample if needed
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            waveform = resampler(waveform)
            sample_rate = sr

        # Convert to mono if requested
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Handle duration and offset
        if duration is not None or offset > 0:
            start_sample = int(offset * sample_rate)
            if duration is not None:
                end_sample = start_sample + int(duration * sample_rate)
                waveform = waveform[:, start_sample:end_sample]
            else:
                waveform = waveform[:, start_sample:]

        return waveform, sample_rate

    except Exception as e:
        # Fallback to librosa for problematic files (e.g., MP3)
        try:
            y, sr_loaded = librosa.load(
                str(path),
                sr=sr,
                mono=mono,
                duration=duration,
                offset=offset
            )
            # librosa returns mono by default, ensure proper shape
            if len(y.shape) == 1:
                y = y[np.newaxis, :]  # Add channel dimension
            waveform = torch.from_numpy(y).float()
            return waveform, sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {path}: {e}, {e2}")


def save_audio(
    waveform: torch.Tensor,
    path: Union[str, Path],
    sr: int = 44100,
    format: Optional[str] = None,
) -> None:
    """
    Save audio tensor to file.

    Args:
        waveform: Audio tensor of shape [channels, samples]
        path: Output file path
        sr: Sample rate
        format: Audio format (default: infer from path extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure waveform is on CPU and proper shape
    if waveform.device != torch.device("cpu"):
        waveform = waveform.cpu()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension

    torchaudio.save(str(path), waveform, sr, format=format)


def get_audio_duration(path: Union[str, Path]) -> float:
    """
    Get audio file duration in seconds without loading the full file.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        info = torchaudio.info(str(path))
        return info.num_frames / info.sample_rate
    except Exception:
        # Fallback to librosa
        return librosa.get_duration(path=str(path))


def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """
    Normalize audio by RMS (Root Mean Square) level.

    Args:
        waveform: Audio tensor of shape [channels, samples]
        target_rms: Target RMS level (default: 0.1)

    Returns:
        RMS-normalized audio tensor
    """
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    if current_rms > 0:
        return waveform * (target_rms / current_rms)
    return waveform


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert mono audio to stereo by duplicating the channel.
    If already stereo or multi-channel, return as-is.

    Args:
        waveform: Audio tensor of shape [channels, samples]

    Returns:
        Stereo audio tensor of shape [2, samples]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)

    return waveform[:2, :]  # Take first 2 channels if > 2


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert stereo or multi-channel audio to mono by averaging channels.

    Args:
        waveform: Audio tensor of shape [channels, samples]

    Returns:
        Mono audio tensor of shape [1, samples]
    """
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)

    if waveform.shape[0] == 1:
        return waveform

    return torch.mean(waveform, dim=0, keepdim=True)
