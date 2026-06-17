"""
FMA-small GenreMaster mastering training.
Uses per-genre LUFS priors from genre_lufs_priors.json to construct
genre-aware (input, target) pairs, giving the model a genuine
genre-correlated loudness signal.

Usage:
    python experiments/fma_mastering/train_mastering.py \
        --config configs/fma_mastering.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from src.data.fma_small import setup_fma_small
from src.data.transforms import GenreAwarePreMasterTransform
from src.models.genremaster import GenreMaster
from src.losses import create_loss_function
from src.utils import get_device, seed_everything


def collate_fn(batch):
    waveforms  = torch.stack([b['waveform']   for b in batch])
    genre_idxs = torch.tensor([b['genre_idx'] for b in batch])
    genres     = [b['genre']                  for b in batch]
    return waveforms, genre_idxs, genres


def train_epoch(model, loader, criterion, optimizer, transform, device, grad_clip, epoch):
    model.train()
    total_loss, n = 0.0, 0
    component_sums = {'loudness': 0., 'spectral': 0., 'dynamic': 0., 'perceptual': 0.}

    for waveforms, genre_idxs, genres in loader:
        # Build (input, target) pairs using genre-aware LUFS degradation
        inputs, targets = [], []
        for i, (wav, genre) in enumerate(zip(waveforms, genres)):
            inp, tgt = transform(wav, genre, training=True)
            inputs.append(inp)
            targets.append(tgt)
        inputs  = torch.stack(inputs).to(device)
        targets = torch.stack(targets).to(device)
        genre_idxs = genre_idxs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, genre_idxs)

        components = criterion(outputs, targets, return_components=True)
        loss = components['total']
        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = len(genres)
        total_loss += loss.item() * bs
        n += bs
        for k in component_sums:
            v = components.get(k, 0.)
            component_sums[k] += (v.item() if hasattr(v, 'item') else float(v)) * bs

    avg = total_loss / max(n, 1)
    avg_components = {k: v / max(n, 1) for k, v in component_sums.items()}
    return avg, avg_components


@torch.no_grad()
def val_epoch(model, loader, criterion, transform, device):
    model.eval()
    total_loss, n = 0.0, 0
    for waveforms, genre_idxs, genres in loader:
        inputs, targets = [], []
        for wav, genre in zip(waveforms, genres):
            inp, tgt = transform(wav, genre, training=False)
            inputs.append(inp)
            targets.append(tgt)
        inputs  = torch.stack(inputs).to(device)
        targets = torch.stack(targets).to(device)
        genre_idxs = genre_idxs.to(device)

        outputs = model(inputs, genre_idxs)
        loss = criterion(outputs, targets)
        if torch.isfinite(loss):
            total_loss += loss.item() * len(genres)
            n += len(genres)

    return total_loss / max(n, 1)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['experiment']['seed'])
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(cfg['output']['checkpoint_dir'])
    log_dir = Path(cfg['output']['log_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Data
    datasets, genre_to_idx = setup_fma_small(
        audio_dir=cfg['data']['audio_dir'],
        metadata_csv=cfg['data']['metadata_csv'],
        sample_rate=cfg['data']['sample_rate'],
        duration=cfg['data']['duration'],
        seed=cfg['experiment']['seed'],
        cache_dir=cfg['data'].get('cache_dir', None),
    )
    print(f"Train: {len(datasets['train'])}  Val: {len(datasets['val'])}  Test: {len(datasets['test'])}")

    train_loader = DataLoader(datasets['train'], batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=cfg['device']['num_workers'],
                              collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(datasets['val'],   batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=cfg['device']['num_workers'],
                              collate_fn=collate_fn)

    # Genre-aware transform
    transform = GenreAwarePreMasterTransform(
        priors_path=cfg['data']['lufs_priors'],
        sample_rate=cfg['data']['sample_rate'],
    )

    # Model
    model = GenreMaster(
        n_genres=cfg['model']['n_genres'],
        encoder_type=cfg['model']['encoder_type'],
        audio_feature_dim=cfg['model']['audio_feature_dim'],
        genre_latent_dim=cfg['model']['genre_latent_dim'],
        sample_rate=cfg['data']['sample_rate'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = create_loss_function(
        sample_rate=cfg['data']['sample_rate'],
        loss_weights=cfg['loss']['weights'],
        use_stft=str(device) == 'cuda',
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
        betas=cfg['optimizer']['betas'],
        eps=cfg['optimizer']['eps'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['num_epochs'] - cfg['training']['warmup_epochs'],
    )

    history = {'train_loss': [], 'val_loss': [], 'train_components': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        train_loss, components = train_epoch(model, train_loader, criterion, optimizer,
                                             transform, device, cfg['training']['grad_clip'], epoch)
        val_loss = val_epoch(model, val_loader, criterion, transform, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_components'].append(components)

        print(f"Epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f} "
              f"| loud={components['loudness']:.3f} spec={components['spectral']:.3f} "
              f"dyn={components['dynamic']:.3f}")

        if val_loss < best_val_loss - cfg['training']['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': cfg,
                'genre_to_idx': genre_to_idx,
            }, out_dir / 'best_model.pt')
            print(f"  --> Saved best (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % cfg['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': cfg,
                'genre_to_idx': genre_to_idx,
            }, out_dir / f'checkpoint_epoch_{epoch}.pt')

    with open(log_dir / 'mastering_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fma_mastering.yaml')
    args = parser.parse_args()
    main(args.config)
