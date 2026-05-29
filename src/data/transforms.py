import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln

class LoudnessNormalize:
    def __init__(self, target_lufs=-18.0, sample_rate=44100):
        self.target_lufs = target_lufs
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(self.sample_rate)

    def __call__(self, waveform):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
            
        # waveform is (channels, samples), pyln expects (samples, channels)
        audio_np = waveform.numpy().T
        
        try:
            loudness = self.meter.integrated_loudness(audio_np)
            if np.isinf(loudness):
                return waveform
                
            audio_normalized = pyln.normalize.loudness(audio_np, loudness, self.target_lufs)
            return torch.from_numpy(audio_normalized.T).to(waveform.dtype)
        except Exception:
            return waveform

class PipelineTransform:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

def create_premaster_transforms(target_lufs, sample_rate, augment_train=False):
    """
    Returns train and validation transforms.
    """
    base_transform = LoudnessNormalize(target_lufs, sample_rate)
    return PipelineTransform([base_transform]), PipelineTransform([base_transform])
