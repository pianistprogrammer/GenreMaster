"""
Preprocess FMA-small audio files into cached .pt tensors.

Run once before training. Saves each track as a mono waveform tensor
at the target sample rate and duration so training never touches MP3 files.

Usage:
    python scripts/preprocess_fma.py --config configs/fma_mastering.yaml
    python scripts/preprocess_fma.py --audio_dir /path/to/fma_small \
        --metadata_csv /path/to/tracks.csv --cache_dir data/fma_cache
"""

import argparse
import warnings
from pathlib import Path

import torch
import torchaudio
import pandas as pd
import yaml
from tqdm import tqdm

FMA_SMALL_GENRES = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock',
]
FMA_CORRUPTED = {98565, 98567, 98569, 99134, 108925, 133297}


def preprocess(audio_dir: Path, metadata_csv: Path, cache_dir: Path,
               sample_rate: int = 22050, duration: float = 30.0):
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_length = int(sample_rate * duration)

    tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
    small = tracks[tracks['set', 'subset'] == 'small'].copy()
    small = small[small['track', 'genre_top'].isin(FMA_SMALL_GENRES)]

    valid_ids = []
    for tid in small.index:
        if tid in FMA_CORRUPTED:
            continue
        tid_str = f"{tid:06d}"
        if (audio_dir / tid_str[:3] / f"{tid_str}.mp3").exists():
            valid_ids.append(tid)

    print(f"Found {len(valid_ids)} valid tracks. Caching to {cache_dir} ...")

    ok, skipped, errors = 0, 0, 0
    for tid in tqdm(valid_ids, ncols=80):
        out_path = cache_dir / f"{tid}.pt"
        if out_path.exists():
            skipped += 1
            continue

        tid_str = f"{tid:06d}"
        mp3_path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                waveform, sr = torchaudio.load(str(mp3_path))

            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if waveform.shape[1] >= target_length:
                start = (waveform.shape[1] - target_length) // 2
                waveform = waveform[:, start:start + target_length]
            else:
                waveform = torch.nn.functional.pad(
                    waveform, (0, target_length - waveform.shape[1]))

            torch.save(waveform, out_path)
            ok += 1
        except Exception as e:
            # Save a zero tensor so the dataloader never hits a missing file
            torch.save(torch.zeros((1, target_length)), out_path)
            errors += 1

    print(f"Done. {ok} processed, {skipped} already cached, {errors} errors (saved as silence).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--metadata_csv', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='data/fma_cache')
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--duration', type=float, default=30.0)
    args = parser.parse_args()

    if args.config:
        cfg = yaml.safe_load(open(args.config))
        data = cfg.get('data', {})
        audio_dir  = Path(args.audio_dir  or data['audio_dir'])
        metadata_csv = Path(args.metadata_csv or data['metadata_csv'])
        sample_rate = data.get('sample_rate', args.sample_rate)
        duration    = data.get('duration',    args.duration)
    else:
        audio_dir    = Path(args.audio_dir)
        metadata_csv = Path(args.metadata_csv)
        sample_rate  = args.sample_rate
        duration     = args.duration

    cache_dir = Path(args.cache_dir)
    preprocess(audio_dir, metadata_csv, cache_dir, sample_rate, duration)


if __name__ == '__main__':
    main()
