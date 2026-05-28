# GenreMaster: Genre-Conditioned Automatic Music Mastering

Implementation of PRD-1 - Genre-conditioned neural network for automatic music mastering using FiLM conditioning.

## Paper
**Working Title**: GenreMaster: A Genre-Conditioned Neural Network for Automatic Music Mastering  
**Target Venue**: IEEE/ACM TASLP  
**PRD**: See `/Users/I558118/Desktop/Prds/PRD_1_Genre_Conditioned_Mastering.md`

## Datasets
- **FMA Medium**: 15,650 tracks for genre embedding training
- **MUSDB18**: 144 stem tracks for evaluation
- **GTZAN**: 1,000 tracks for validation

## System Architecture
1. Genre Embedding Network (contrastive pre-training on FMA)
2. FiLM Conditioning Layer
3. Differentiable Mastering Chain (DASP-based)
   - Loudness normalization
   - Parametric EQ (8-band)
   - Multiband compression
   - True peak limiter
   - Stereo width control

## Hardware
- Apple Silicon M4 48GB (MPS backend)
- Training: ~48-72 GPU-hours estimated

## Installation
```bash
cd ~/Documents/Projects/GenreMaster
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage
```bash
# Phase 1: Extract FMA features
python experiments/01_extract_fma_features.py

# Phase 2: Train genre embeddings
python experiments/02_train_genre_embeddings.py

# Phase 3: Train mastering chain
python experiments/03_train_mastering_chain.py

# Phase 4: Evaluate on MUSDB18
python experiments/04_evaluate.py
```
