# GenreMaster: Genre-Conditioned Automatic Music Mastering

Implementation of PRD-1 - Genre-conditioned neural network for automatic music mastering using FiLM conditioning.

**Status**: ✅ **Implementation Complete - Ready for Training**

---

## Quick Start

```bash
# Run end-to-end pipeline test (validates everything works)
uv run python tests/test_pipeline.py

# Start training (quick test: 5 epochs, 50 samples)
uv run python experiments/run_main.py --config configs/default.yaml
```

---

## Project Overview

### Architecture
```
Input Audio → Spectral Encoder → Genre Embedding (z_g)
                                        ↓
                                  FiLM Conditioning
                                        ↓
                              DSP Parameter Predictor
                                        ↓
                           Differentiable Mastering Chain
                                        ↓
                                Mastered Output
```

**Key Components**:
- **Spectral Encoder**: ResNet-18 or lightweight CNN on log-mel spectrograms
- **Genre Embedding**: 16 genres → 128-dim latent with contrastive learning
- **FiLM Conditioning**: Genre-modulated processing
- **DSP Chain**: Loudness norm, 8-band EQ, multiband compression, limiter, stereo width

**Model Size**: 2.5M parameters (genre conditioning adds 531k params)

---

## Installation

```bash
# Clone and setup
cd /Users/I558118/Documents/Projects/GenreMaster
uv sync

# Datasets (already on disk)
# - FMA Medium: /Volumes/LLModels/Datasets/fma_medium
# - MUSDB18-HQ: /Volumes/LLModels/Datasets/musdb18
```

---

## Training Pipeline

### 1. Genre Embedding Pre-training (Optional)
```bash
uv run python experiments/run_pretrain.py --config configs/pretrain.yaml
```
- Pre-trains genre embeddings with contrastive loss
- 20 epochs, 200 samples (~2 hours on M4 Max)
- Saves: `results/pretraining/genre_embedding_pretrained.pt`

### 2. Main Training
```bash
uv run python experiments/run_main.py --config configs/default.yaml
```
- Full GenreMaster training
- Default: 50 epochs, 100 samples (~4-6 hours)
- Checkpoints every 5 epochs
- Trackio logging enabled

**Resume from checkpoint:**
```bash
uv run python experiments/run_main.py --config configs/default.yaml --resume results/checkpoints/best_model.pt
```

### 3. Ablation Study
```bash
uv run python experiments/run_ablation.py --config configs/ablation.yaml
```
Compares 4 variants:
- Baseline (unconditioned)
- Full conditioned
- ResNet encoder
- Genre-only mode

Results saved to: `results/ablation/comparison.csv`

### 4. Evaluation
```bash
uv run python experiments/run_evaluation.py --checkpoint results/checkpoints/best_model.pt
```
- Evaluates on MUSDB18-HQ test set
- Compares: Unprocessed, Unconditioned, Full GenreMaster
- Metrics: LUFS MAE, LRA MAE, True Peak MAE, Spectral Tilt MAE

---

## Configuration

Edit `configs/default.yaml` to adjust:

```yaml
data:
  n_train_samples: 100  # Set to null for full dataset
  n_val_samples: 20

training:
  batch_size: 4
  num_epochs: 50  # Increase to 100 for full training
  learning_rate: 1.0e-4

loss:
  weights:
    loudness: 1.0
    spectral: 0.5
    dynamic: 0.5
    perceptual: 0.1

device:
  type: "auto"  # auto-detect MPS/CUDA/CPU
```

---

## Testing

```bash
# Core utilities
uv run python tests/test_utils.py

# Feature extraction
uv run python tests/test_features.py

# Full model
uv run python tests/test_genremaster.py

# End-to-end pipeline (CRITICAL before training)
uv run python tests/test_pipeline.py
```

All tests use **real FMA audio files** (no synthetic data).

---

## Datasets

### FMA Medium (Training)
- **Path**: `/Volumes/LLModels/Datasets/fma_medium`
- **Tracks**: 5,689 valid MP3 files
- **Genres**: 16 (Rock, Electronic, Jazz, Classical, Pop, etc.)
- **Splits**: 4,543 train / 562 val / 584 test

### MUSDB18-HQ (Evaluation)
- **Path**: `/Volumes/LLModels/Datasets/musdb18`
- **Tracks**: 50 test tracks (stems available)
- **Use**: Final evaluation and baseline comparisons

---

## Results & Outputs

```
results/
├── checkpoints/           # Model checkpoints (.pt files)
│   ├── best_model.pt     # Best validation loss
│   └── checkpoint_epoch_*.pt
├── logs/                 # Trackio logs
├── audio_samples/        # Sample outputs during training
├── ablation/             # Ablation study results
│   └── comparison.csv
└── evaluation/           # MUSDB18-HQ evaluation
    ├── comparison.csv
    └── detailed_results.json
```

---

## Device Support

- ✅ **MPS** (Apple Silicon) - tested on M4 Max
- ✅ **CUDA** (NVIDIA GPUs) - ready, not yet tested
- ✅ **CPU** - fallback supported

Device auto-detected via `get_device()` utility.

---

## Key Features

✅ **Real Data Testing**: All tests use actual FMA audio (no synthetic data)  
✅ **Resume Support**: All experiments checkpoint and resume automatically  
✅ **Trackio Integration**: Metrics logged for visualization  
✅ **Device Agnostic**: Works on MPS, CUDA, or CPU  
✅ **Modular**: Easy to swap encoders, ablate components  
✅ **Validated**: End-to-end pipeline tested with 50 real samples  

---

## Research Paper (PRD)

See: `PRD_1_Genre_Conditioned_Mastering.md`

**Target Venue**: IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

**Contributions**:
1. First genre-conditioned differentiable mastering network
2. FiLM-based conditioning for DSP parameter prediction
3. Genre-stratified FMA mastering signature extraction

---

## Citation

```bibtex
@article{genremaster2026,
  title={GenreMaster: A Genre-Conditioned Neural Network for Automatic Music Mastering},
  author={[Author]},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2026}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Implementation Status

- [x] Core infrastructure (utils, data loaders, features)
- [x] Model architecture (encoder, genre embedding, DSP, FiLM)
- [x] Loss functions (4 components)
- [x] Training scripts (main, pretrain, ablation, evaluation)
- [x] Tests (all passing with real data)
- [x] End-to-end pipeline validated
- [ ] Genre embedding pre-training (ready to run)
- [ ] Main model training (ready to run)
- [ ] Ablation study (ready to run)
- [ ] MUSDB18-HQ evaluation (ready to run)

**Next**: Run training experiments! 🚀
