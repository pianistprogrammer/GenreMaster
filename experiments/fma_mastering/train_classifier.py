"""
FMA-small genre classifier training.
Trains a ResNet-based genre classifier on the 8 balanced FMA-small genres.
Usage:
    python experiments/fma_mastering/train_classifier.py \
        --config configs/fma_classifier.yaml
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

from src.data.fma_small import setup_fma_small, FMA_SMALL_GENRES
from src.models.resnet_genre_classifier import ResNetGenreClassifier
from src.utils import get_device, seed_everything


def collate_fn(batch):
    waveforms = torch.stack([b['waveform'] for b in batch])
    genre_idxs = torch.tensor([b['genre_idx'] for b in batch])
    return waveforms, genre_idxs


def train_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        logits = model(waveforms)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


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
    )
    print(f"Train: {len(datasets['train'])}  Val: {len(datasets['val'])}  Test: {len(datasets['test'])}")
    print(f"Genres: {list(genre_to_idx.keys())}")

    train_loader = DataLoader(datasets['train'], batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=cfg['device']['num_workers'],
                              collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(datasets['val'],   batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=cfg['device']['num_workers'],
                              collate_fn=collate_fn)

    # Model
    model = ResNetGenreClassifier(
        n_genres=cfg['model']['n_genres'],
        n_mels=cfg['model']['n_mels'],
        n_fft=cfg['model']['n_fft'],
        hop_length=cfg['model']['hop_length'],
        sample_rate=cfg['data']['sample_rate'],
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['model']['dropout'],
        spec_augment=cfg['model']['spec_augment'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
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

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'best_val_acc': 0.0}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                            cfg['training']['grad_clip'])
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc*100:.1f}% "
              f"| val_loss={val_loss:.4f} acc={val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc * 100
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc, 'config': cfg, 'genre_to_idx': genre_to_idx},
                       out_dir / 'best_classifier.pt')
            print(f"  --> New best: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(log_dir / 'classifier_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to {out_dir / 'best_classifier.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fma_classifier.yaml')
    args = parser.parse_args()
    main(args.config)
