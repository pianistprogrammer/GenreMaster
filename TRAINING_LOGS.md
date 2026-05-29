# Training Logs and Metrics

## Overview

Training scripts now save comprehensive JSON logs for later analysis and visualization.

---

## JSON Files Saved

### 1. Main Training: `results/logs/training_history.json`

**Location**: `results/logs/training_history.json`

**Structure**:
```json
{
  "config": { ... },                    // Full config used for training
  "experiment_name": "genremaster-main",
  "start_time": "2026-05-29 12:00:00",
  "end_time": "2026-05-29 16:30:00",
  "device": "mps:0",
  "best_val_loss": 2.345,
  "total_epochs": 50,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 5.234,
      "train_loss_components": {
        "loudness": 1.234,
        "spectral": 2.345,
        "dynamic": 1.456,
        "perceptual": 0.199
      },
      "val_loss": 4.567,
      "val_loss_components": {
        "loudness": 1.123,
        "spectral": 2.234,
        "dynamic": 1.012,
        "perceptual": 0.198
      },
      "learning_rate": 0.0001,
      "epoch_time_seconds": 235.67,
      "is_best": true
    },
    ...
  ]
}
```

### 2. Pre-training: `results/logs/pretrain_history.json`

**Location**: `results/logs/pretrain_history.json`

**Structure**:
```json
{
  "config": { ... },
  "experiment_name": "genremaster-pretrain",
  "start_time": "2026-05-29 10:00:00",
  "end_time": "2026-05-29 12:00:00",
  "device": "mps:0",
  "best_val_loss": 1.234,
  "total_epochs": 20,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 3.456,
      "train_loss_components": {
        "contrastive": 2.345,
        "classification": 1.111
      },
      "val_loss": 2.789,
      "val_loss_components": {
        "contrastive": 1.789,
        "classification": 1.000
      },
      "val_accuracy": 45.67,
      "learning_rate": 0.0001,
      "epoch_time_seconds": 120.45,
      "is_best": true
    },
    ...
  ]
}
```

---

## Metrics Tracked

### Main Training Metrics
- **Total Loss**: Weighted combination of all loss components
- **Loss Components**:
  - `loudness`: LUFS (integrated loudness) matching loss
  - `spectral`: Log-mel spectrogram similarity loss
  - `dynamic`: Dynamic range (LRA + crest factor) matching loss
  - `perceptual`: Multi-resolution STFT perceptual quality loss
- **Learning Rate**: Current optimizer learning rate
- **Epoch Time**: Wall-clock time per epoch in seconds
- **Is Best**: Boolean flag indicating if this epoch achieved best validation loss

### Pre-training Metrics
- **Total Loss**: Weighted combination of contrastive + classification
- **Loss Components**:
  - `contrastive`: InfoNCE contrastive loss for genre separation
  - `classification`: Cross-entropy loss for genre classification
- **Validation Accuracy**: Genre classification accuracy on validation set (%)
- **Learning Rate**: Current optimizer learning rate
- **Epoch Time**: Wall-clock time per epoch in seconds
- **Is Best**: Boolean flag indicating if this epoch achieved best validation loss

---

## Usage Examples

### Load and Plot Training History

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('results/logs/training_history.json', 'r') as f:
    history = json.load(f)

# Extract metrics
epochs = [e['epoch'] for e in history['epochs']]
train_losses = [e['train_loss'] for e in history['epochs']]
val_losses = [e['val_loss'] for e in history['epochs']]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GenreMaster Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
```

### Analyze Loss Components

```python
import pandas as pd

# Convert to DataFrame for easy analysis
epochs_data = history['epochs']
df = pd.DataFrame(epochs_data)

# Expand loss components
train_components = pd.DataFrame(df['train_loss_components'].tolist())
train_components.columns = ['train_' + c for c in train_components.columns]

val_components = pd.DataFrame(df['val_loss_components'].tolist())
val_components.columns = ['val_' + c for c in val_components.columns]

# Combine
full_df = pd.concat([df[['epoch', 'train_loss', 'val_loss']], 
                     train_components, val_components], axis=1)

# Analyze
print(full_df.describe())
print(f"\nBest epoch: {full_df.loc[full_df['val_loss'].idxmin(), 'epoch']}")
print(f"Best val loss: {full_df['val_loss'].min():.6f}")
```

### Find Best Checkpoint

```python
# Find epoch with best validation loss
best_epoch_data = min(history['epochs'], key=lambda x: x['val_loss'])

print(f"Best Epoch: {best_epoch_data['epoch']}")
print(f"Val Loss: {best_epoch_data['val_loss']:.6f}")
print(f"Loss Components:")
for component, value in best_epoch_data['val_loss_components'].items():
    print(f"  {component}: {value:.6f}")
```

### Compare Multiple Runs

```python
import glob

# Load all training histories
runs = []
for path in glob.glob('results/logs/training_history_*.json'):
    with open(path) as f:
        runs.append(json.load(f))

# Compare best validation losses
for run in runs:
    print(f"{run['experiment_name']}: {run['best_val_loss']:.6f}")
```

---

## Additional Logs

### Trackio Real-time Logs
- **Location**: Trackio cloud dashboard
- **Usage**: Real-time monitoring during training
- **Metrics**: Same as JSON logs, but streamed in real-time

### Model Checkpoints
- **Location**: `results/checkpoints/`
- **Files**:
  - `best_model.pt` - Best validation loss checkpoint
  - `final_model.pt` - Final epoch checkpoint
  - `checkpoint_epoch_N.pt` - Periodic checkpoints every N epochs
- **Contains**: Model weights, optimizer state, scheduler state, epoch info, config

### Audio Samples (Optional)
- **Location**: `results/audio_samples/`
- **Usage**: Sample mastered outputs saved periodically for listening
- **Frequency**: Configurable via `save_audio_every` in config

---

## Visualization Scripts

See `scripts/plot_training.py` (coming soon) for ready-made visualization utilities.

---

## Notes

- JSON files are updated **after each epoch**, so you can monitor progress even during training
- All timestamps use local system time
- Loss values are averaged over all batches in an epoch
- The `is_best` flag helps quickly identify which epochs improved validation performance
- History files survive crashes - if training is interrupted, the JSON will contain all completed epochs
