"""
FMA-small GenreMaster evaluation.
Loads the best checkpoint, runs inference on the test set,
computes per-genre DSP parameters and reconstruction metrics,
and saves results to results/fma/.

Usage:
    python experiments/fma_mastering/evaluate.py \
        --checkpoint results/fma/checkpoints/best_model.pt \
        --config    configs/fma_mastering.yaml
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from src.data.fma_small import setup_fma_small, FMA_SMALL_GENRES
from src.data.transforms import GenreAwarePreMasterTransform
from src.losses import create_loss_function
from src.models.genremaster import GenreMaster
from src.utils import get_device, seed_everything


def collate_fn(batch):
    waveforms  = torch.stack([b['waveform']   for b in batch])
    genre_idxs = torch.tensor([b['genre_idx'] for b in batch])
    genres     = [b['genre']                  for b in batch]
    return waveforms, genre_idxs, genres


@torch.no_grad()
def run_evaluation(model, loader, criterion, transform, device):
    model.eval()
    genre_params   = defaultdict(lambda: defaultdict(list))
    genre_losses   = defaultdict(list)
    all_snr, all_corr = [], []

    for waveforms, genre_idxs, genres in loader:
        inputs, targets = [], []
        for wav, genre in zip(waveforms, genres):
            inp, tgt = transform(wav, genre, training=False)
            inputs.append(inp)
            targets.append(tgt)
        inputs  = torch.stack(inputs).to(device)
        targets = torch.stack(targets).to(device)
        genre_idxs = genre_idxs.to(device)

        outputs, params = model(inputs, genre_idxs, return_params=True)

        for i, genre in enumerate(genres):
            loss = criterion(outputs[i:i+1], targets[i:i+1]).item()
            genre_losses[genre].append(loss)

            genre_params[genre]['target_lufs'].append(params['target_lufs'][i].item())
            genre_params[genre]['eq_gains'].append(params['eq_gains'][i].tolist())
            genre_params[genre]['comp_thresholds'].append(params['comp_thresholds'][i].tolist())
            genre_params[genre]['comp_ratios'].append(params['comp_ratios'][i].tolist())
            genre_params[genre]['stereo_width'].append(params['stereo_width'][i].item())

            # SNR and correlation
            signal = targets[i].cpu()
            noise  = (outputs[i] - targets[i]).cpu()
            snr = 10 * np.log10(
                signal.pow(2).mean().item() / max(noise.pow(2).mean().item(), 1e-10)
            )
            corr = np.corrcoef(
                signal.flatten().numpy(), outputs[i].cpu().flatten().numpy()
            )[0, 1]
            all_snr.append(snr)
            all_corr.append(float(corr))

    # Aggregate per genre
    per_genre = {}
    for genre in FMA_SMALL_GENRES:
        if genre not in genre_losses:
            continue
        p = genre_params[genre]
        per_genre[genre] = {
            'test_loss':       float(np.mean(genre_losses[genre])),
            'n':               len(genre_losses[genre]),
            'target_lufs':     float(np.mean(p['target_lufs'])),
            'eq_gains':        list(np.mean(p['eq_gains'],         axis=0)),
            'comp_thresholds': list(np.mean(p['comp_thresholds'],  axis=0)),
            'comp_ratios':     list(np.mean(p['comp_ratios'],      axis=0)),
            'stereo_width':    float(np.mean(p['stereo_width'])),
        }

    return {
        'test_loss':   float(np.mean([l for ls in genre_losses.values() for l in ls])),
        'snr_db':      float(np.mean(all_snr)),
        'correlation': float(np.mean(all_corr)),
        'per_genre':   per_genre,
    }


def main(checkpoint_path: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['experiment']['seed'])
    device = get_device()

    out_dir = Path(cfg['output']['checkpoint_dir']).parent / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, genre_to_idx = setup_fma_small(
        audio_dir=cfg['data']['audio_dir'],
        metadata_csv=cfg['data']['metadata_csv'],
        sample_rate=cfg['data']['sample_rate'],
        duration=cfg['data']['duration'],
    )
    test_loader = DataLoader(datasets['test'], batch_size=4, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    print(f"Test set: {len(datasets['test'])} clips")

    transform = GenreAwarePreMasterTransform(
        priors_path=cfg['data']['lufs_priors'],
        sample_rate=cfg['data']['sample_rate'],
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = GenreMaster(
        n_genres=cfg['model']['n_genres'],
        encoder_type=cfg['model']['encoder_type'],
        audio_feature_dim=cfg['model']['audio_feature_dim'],
        genre_latent_dim=cfg['model']['genre_latent_dim'],
        sample_rate=cfg['data']['sample_rate'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['best_val_loss']:.4f})")

    criterion = create_loss_function(
        sample_rate=cfg['data']['sample_rate'],
        loss_weights=cfg['loss']['weights'],
    )

    results = run_evaluation(model, test_loader, criterion, transform, device)

    print(f"\nTest loss:   {results['test_loss']:.4f}")
    print(f"SNR:         {results['snr_db']:.2f} dB")
    print(f"Correlation: {results['correlation']:.4f}")
    print("\nPer-genre test loss:")
    for genre, r in results['per_genre'].items():
        print(f"  {genre:15s}: {r['test_loss']:.4f}  (n={r['n']})")

    out_path = out_dir / 'evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/fma/checkpoints/best_model.pt')
    parser.add_argument('--config',     default='configs/fma_mastering.yaml')
    args = parser.parse_args()
    main(args.checkpoint, args.config)
