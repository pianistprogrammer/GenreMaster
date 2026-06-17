import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
import warnings
import json
from pathlib import Path

class LoudnessNormalize:
    def __init__(self, target_lufs=-18.0, sample_rate=44100):
        self.target_lufs = target_lufs
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(self.sample_rate)

    def __call__(self, waveform):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
            
        # waveform is (channels, samples), pyln expects (samples, channels)
        # Ensure it is on the CPU before converting to NumPy
        audio_np = waveform.detach().cpu().numpy().T
        
        try:
            loudness = self.meter.integrated_loudness(audio_np)
            if np.isinf(loudness):
                return waveform
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                audio_normalized = pyln.normalize.loudness(audio_np, loudness, self.target_lufs)
                
            # Prevent excessive clipping issues from flowing to model output explicitly
            audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
            
            # Ensure the output tensor is placed on the original device and matches dtype
            return torch.from_numpy(audio_normalized.T).to(device=waveform.device, dtype=waveform.dtype)
        except Exception:
            return waveform

class PreMasterTransform:
    def __init__(self, target_lufs, sample_rate, augment=False):
        self.target_lufs = target_lufs
        self.sample_rate = sample_rate
        self.augment = augment
        self.normalizer = LoudnessNormalize(target_lufs, sample_rate)
        
    def __call__(self, waveform):
        target = waveform.clone()
        pre_master = self.normalizer(waveform)
        # Pre-master and target pairs are expected
        return pre_master, target

def create_premaster_transforms(target_lufs, sample_rate, augment_train=False):
    """
    Returns train and validation transforms.
    """
    train_transform = PreMasterTransform(target_lufs, sample_rate, augment=augment_train)
    val_transform = PreMasterTransform(target_lufs, sample_rate, augment=False)
    return train_transform, val_transform


class GenreAwarePreMasterTransform:
    """
    Creates (input, target) pairs using per-genre LUFS priors measured from
    FMA-small masters. The input is normalized to the genre mean LUFS; the
    target is the original clip. This gives the model a genre-correlated
    loudness signal to learn from, unlike the flat -18 LUFS protocol.

    priors_path: path to genre_lufs_priors.json produced by measure_lufs.py
    noise_std: std of Gaussian noise added to LUFS target during training
               to prevent the model from memorising a single value per genre
    """

    def __init__(self, priors_path: str, sample_rate: int = 22050, noise_std: float = 1.5):
        self.sample_rate = sample_rate
        self.noise_std = noise_std

        with open(priors_path) as f:
            priors = json.load(f)

        self.genre_lufs: dict = {g: p['mean_lufs'] for g, p in priors.items()}
        self.meter = pyln.Meter(sample_rate)

    def __call__(self, waveform: torch.Tensor, genre: str, training: bool = True):
        """
        Args:
            waveform: [channels, samples] original (mastered) clip
            genre: genre string matching keys in priors JSON
            training: if True, adds LUFS jitter for regularisation
        Returns:
            (input_wav, target_wav) both as torch.Tensor [channels, samples]
        """
        target = waveform.clone()

        target_lufs = self.genre_lufs.get(genre, -18.0)
        if training and self.noise_std > 0:
            target_lufs += np.random.normal(0, self.noise_std)
        target_lufs = float(np.clip(target_lufs, -30.0, -8.0))

        normalizer = LoudnessNormalize(target_lufs, self.sample_rate)
        degraded = normalizer(waveform)

        return degraded, target

