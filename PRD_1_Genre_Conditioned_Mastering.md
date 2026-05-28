# PRD-1: Genre-Conditioned Automatic Mastering via Latent Conditioning
**Research Product Requirements Document**

| Field | Value |
|---|---|
| **Working Title** | GenreMaster: A Genre-Conditioned Neural Network for Automatic Music Mastering |
| **Target Venue** | IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP) |
| **Submission Type** | Regular Paper (6,000–8,000 words) |
| **Open Access** | Hybrid (IEEE OA option available) |
| **Datasets** | FMA Large, MUSDB18-HQ, MoisesDB |
| **No Human Subjects** | Yes — purely audio dataset experiments |

---

## 1. Research Problem

### 1.1 Motivation

Automatic music mastering — the final stage of music production that optimises loudness, dynamic range, tonal balance, and stereo width for distribution — remains an open research problem. Current learned approaches treat mastering either as a genre-agnostic signal transformation or as pure reference-based style transfer: given a reference track, reproduce its production characteristics on a new input.

Both paradigms have a fundamental flaw: **mastering is genre-normative**. Pop tracks target −8 to −10 LUFS integrated loudness, with high limiting and compressed dynamics (LRA 4–8 LU). Classical and jazz recordings are mastered to −18 LUFS or below, preserving wide dynamic range (LRA 15–20 LU). Electronic dance music uses aggressive spectral tilt with boosted sub-bass. These are not stylistic choices — they are genre conventions that professional mastering engineers apply systematically. No existing model encodes this knowledge as a learnable, controllable latent.

### 1.2 The Gap

| Existing Approach | Limitation |
|---|---|
| LANDR / iZotope Ozone (rule-based) | No learning; fixed heuristics per genre |
| Reference-based mastering (Steinmetz et al. 2022) | Requires a reference track at inference; cannot generalise to new genres |
| End-to-end remastering (Lee et al. 2022, arXiv:2202.08520) | Input must be already-processed; no genre conditioning |
| SonicMaster (2025, arXiv:2508.03448) | All-in-one restoration; no explicit genre-conditioned mastering pathway |

**No published work** uses genre as a learnable latent conditioning vector to guide differentiable mastering chains. This is the primary contribution.

---

## 2. Research Objectives

1. **RO-1** — Demonstrate that genre-conditioned mastering outperforms genre-agnostic baselines on standard loudness and dynamic range metrics (LUFS MAE, LRA MAE, spectral tilt error).
2. **RO-2** — Show that genre conditioning enables controlled, interpretable mastering decisions (ablation: conditioning on/off, per-effect).
3. **RO-3** — Demonstrate cross-genre generalisation: a model trained on FMA genres that has never seen a specific genre sub-class can still outperform unconditioned baselines on that sub-class.
4. **RO-4** — Validate on established benchmarks (MUSDB18-HQ, MoisesDB) with direct numeric comparison to published baselines.

---

## 3. Proposed System Architecture

### 3.1 Overview

```
Input Audio (wet/dry mix)
        │
        ▼
  Spectral Encoder (CNN/Transformer)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Genre Embedding Network         Audio Feature Extractor
  (FMA genre labels → latent z_g)  (LUFS, LRA, tilt, width)
        │                              │
        └──────────┬───────────────────┘
                   ▼
       Conditioning Fusion (FiLM layer)
                   │
                   ▼
     Differentiable Mastering Chain
     ┌─────────────────────────────┐
     │  1. Loudness Normalisation  │
     │  2. Parametric EQ (8-band)  │
     │  3. Multiband Compression   │
     │  4. True Peak Limiter       │
     │  5. Stereo Width Control    │
     └─────────────────────────────┘
                   │
                   ▼
           Mastered Output Audio
```

### 3.2 Key Components

**Genre Embedding Network**
- Input: genre label (FMA taxonomy, 161 genres) + audio clip
- Architecture: Linear embedding layer + audio-conditioned refinement (contrastive pre-training on FMA)
- Output: latent vector z_g ∈ ℝ^128

**FiLM Conditioning**
- Feature-wise Linear Modulation (Perez et al. 2018) injects z_g into each layer of the mastering chain
- Allows genre to modulate both scale and shift of processing parameters
- Fully differentiable end-to-end

**Differentiable Mastering Chain**
- Built on `dasp-pytorch` (differentiable audio signal processing)
- Each processor (EQ, compression, limiter, width) is differentiable w.r.t. its parameters
- Parameters predicted by a lightweight MLP conditioned on z_g + audio features

### 3.3 Training Objective

```
L_total = λ_1 · L_loudness + λ_2 · L_spectral + λ_3 · L_dynamic + λ_4 · L_perceptual

Where:
  L_loudness  = MSE(LUFS_pred, LUFS_target)
  L_spectral  = L1(log_mel_spectrogram(pred), log_mel_spectrogram(target))
  L_dynamic   = MSE(LRA_pred, LRA_target) + MSE(crest_factor_pred, crest_factor_target)
  L_perceptual = STFT multi-resolution loss (Yamamoto et al. 2020)
```

---

## 4. Datasets

### 4.1 Training Data

| Dataset | Tracks | Use | Access |
|---|---|---|---|
| **FMA Large** | 106,000 tracks | Genre embedding pre-training; mastering target feature extraction | Free, CC-licensed |
| **FMA Medium** | 25,000 tracks (balanced) | Primary training split | Free, CC-licensed |

**FMA pre-processing pipeline:**
1. Extract per-track mastering signatures: integrated LUFS, LRA, true peak, spectral centroid, spectral tilt (α), M-S ratio, dynamic range (DR14 metric)
2. These become soft regression targets for each genre class
3. Genre-stratified split: 80/10/10 train/val/test, balanced across top-16 genre classes

### 4.2 Evaluation Data

| Dataset | Tracks | Use |
|---|---|---|
| **MUSDB18-HQ** | 150 tracks (100 train / 50 test) | Primary evaluation: mixed stems → mastered reference |
| **MoisesDB** | 240 tracks, 14+ hrs | Extended evaluation: commercial-grade mastered targets |

**Target construction for MUSDB18-HQ:**
- The already-mastered mixture track serves as the mastering target
- Stems are mixed to a −18 LUFS "pre-master" and fed as input
- Ground truth = original mastered mixture

---

## 5. Baselines and Comparisons

All comparisons are on the **MUSDB18-HQ test set (50 tracks)** and **MoisesDB test split**, matching dataset and evaluation conditions of published work.

| Baseline | Reference | Why Included |
|---|---|---|
| **Unprocessed (pre-master)** | — | Lower bound |
| **LANDR** (commercial, rule-based) | Sterne & Razlogova 2019 | Industry reference |
| **Reference-based mastering (DMFX)** | Steinmetz et al. ISMIR 2022 | Best published learned baseline |
| **End-to-end remastering** | Lee et al. arXiv:2202.08520 | Direct architecture predecessor |
| **SonicMaster** | arXiv:2508.03448, 2025 | Most recent all-in-one system |
| **GenreMaster (ours) — no conditioning** | Ablation | Isolates effect of genre conditioning |
| **GenreMaster (ours) — full** | This work | Full proposed system |

---

## 6. Evaluation Metrics

All metrics computed on held-out test set. No new metrics introduced — all are established in the literature.

| Metric | Description | Baseline Comparison |
|---|---|---|
| **LUFS MAE** | Mean absolute error of integrated loudness (LUFS) | vs. all baselines |
| **LRA MAE** | Mean absolute error of loudness range | vs. all baselines |
| **True Peak Error (dBTP)** | Max true peak deviation from target | vs. all baselines |
| **Spectral Tilt Error** | L1 distance of log-spectral slope | vs. all baselines |
| **SI-SDR (dB)** | Scale-invariant signal-to-distortion ratio | vs. all baselines |
| **Multi-resolution STFT loss** | Perceptual audio quality | vs. neural baselines |
| **FAD (Fréchet Audio Distance)** | Distribution-level quality | vs. neural baselines |

### 6.1 Ablation Study Design

| Model Variant | Purpose |
|---|---|
| No genre conditioning | Quantify contribution of z_g |
| Hard genre label (one-hot) vs. learned embedding | Embedding vs. classification |
| Per-effect conditioning vs. global conditioning | FiLM placement analysis |
| Genre seen during training vs. held-out genre | Generalisation measurement |

---

## 7. Scientific Contributions

1. **Primary** — First genre-conditioned differentiable mastering network; measurable improvement over unconditioned and reference-based baselines on MUSDB18-HQ and MoisesDB
2. **Secondary** — FiLM-based conditioning mechanism for audio signal processing chains (transferable to other production tasks)
3. **Tertiary** — Genre-stratified FMA mastering signature extraction (reproducible, shared as supplementary data)

---

## 8. Expected Results and Claims

| Claim | Expected Δ | Measured Against |
|---|---|---|
| GenreMaster outperforms unconditioned baseline | LUFS MAE ↓ ≥15%, LRA MAE ↓ ≥10% | Ablation: no conditioning |
| GenreMaster outperforms reference-based mastering (no reference at inference) | Competitive within ±1 LUFS MAE | Steinmetz et al. 2022 |
| Genre conditioning improves generalisation to held-out genre sub-classes | SDR ↑, FAD ↓ | Cross-genre split experiment |
| FiLM conditioning is better than hard one-hot genre label | All metrics | Ablation |

---

## 9. Implementation Plan

| Phase | Task | Duration |
|---|---|---|
| **P1** | FMA feature extraction pipeline; genre embedding pre-training | 3 weeks |
| **P2** | Differentiable mastering chain implementation (dasp-pytorch) | 2 weeks |
| **P3** | FiLM integration; end-to-end training on FMA Medium | 4 weeks |
| **P4** | Evaluation on MUSDB18-HQ and MoisesDB; baseline comparisons | 3 weeks |
| **P5** | Ablation studies | 2 weeks |
| **P6** | Paper writing, supplementary material, code release | 4 weeks |
| **Total** | | ~18 weeks |

### 9.1 Compute Requirements
- Training: ~48–72 GPU-hours (A100 or equivalent); feasible on M4 with MPS backend for prototyping
- FMA feature extraction: CPU-parallel, ~6 hours for full 106k tracks
- Inference: real-time capable (< 1s per track on CPU)

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| FMA genre labels are noisy (161 classes, hierarchical) | Use top-8 coarse genre classes; verify with audio feature clustering |
| MUSDB18-HQ mastered targets are genre-homogeneous (mostly rock/pop) | Supplement with MoisesDB for genre diversity in evaluation |
| Paired dry/wet data is scarce | Use FMA pre-master simulation (normalise to −18 LUFS) as input |
| Genre conditioning collapses to genre classification | Regularise with reconstruction loss; monitor z_g diversity |

---

## 11. Related Work (Direct Citations)

- Martínez-Ramírez et al. (ISMIR 2022) — automatic mixing with out-of-domain data
- Steinmetz et al. (ICASSP 2021) — differentiable mixing console
- Lee et al. (arXiv 2022) — end-to-end remastering, self-supervised + adversarial
- SonicMaster (arXiv 2025) — all-in-one controllable restoration/mastering
- Perez et al. (2018) — FiLM: visual reasoning with a general conditioning layer
- Sterne & Razlogova (2019) — LANDR and AI mastering platformisation

---

## 12. Code and Reproducibility Plan

- All code released on GitHub under MIT license
- FMA feature extraction script shared as standalone tool
- Pre-trained genre embeddings released on HuggingFace
- Evaluation scripts reproduce all table results from a single shell command
- Docker container for full pipeline reproducibility
