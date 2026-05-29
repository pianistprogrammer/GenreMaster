import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
import warnings

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
