# Adaptation Depth in Self-Supervised Speech Emotion Recognition (SER)

This repository contains the official code, nested 6-fold speaker-independent splits, experiment configurations, per-utterance predictions, and statistical analysis scripts for studying **adaptation depth** in self-supervised speech representations (`wav2vec2-base` and `WavLM Base+`) on **RAVDESS** and zero-shot cross-corpus transfer to **CREMA-D**.

---

## Key Findings

1. **Adapting deeper representations significantly outperforms frozen features:**
   - On 6-fold speaker-independent nested CV on RAVDESS (1,440 speech utterances, 8 emotions, 24 actors), `wav2vec2 Partial FT (L8-L11)` reaches **72.37 ± 6.89% UAR** and `wav2vec2 Full FT` reaches **71.55 ± 6.11% UAR** (3-seed fold averages), compared to **56.64 ± 5.90% UAR** for `wav2vec2 Weighted Sum` (+14.91 pp, 95% CI `[+3.31, +26.51]`, paired *t*-test *p* = 0.0214) and **42.19 ± 5.50% UAR** for `wav2vec2 Linear Probe`.
2. **Partial fine-tuning (top 4 transformer layers) matches full fine-tuning on clean speech:**
   - `wav2vec2 Full FT` vs. `wav2vec2 Partial FT (L8-L11)` differs by `-0.83 pp` UAR (95% CI `[-4.40, +2.75]`, *p* = 0.5793), indicating no statistically significant clean-speech advantage for unfreezing the CNN feature encoder and lower transformer layers.
3. **Ablations show no significant effect from cropping strategy or class weighting:**
   - **Center crop vs. random crop:** `-0.37 pp` UAR (95% CI `[-5.31, +4.57]`, *p* = 0.8556).
   - **Weighted CE vs. unweighted CE:** `-0.02 pp` UAR (95% CI `[-4.22, +4.18]`, *p* = 0.9899), with `0 / 18` training collapses in both conditions.
4. **Noise robustness (fixed-seed white and babble noise at 20, 10, and 0 dB SNR):**
   - `wav2vec2 Full FT` achieves higher point-estimate UAR than `wav2vec2 Partial FT` under additive white noise (+4.21 pp at 20 dB, +6.25 pp at 10 dB, +7.51 pp at 0 dB) and babble noise (+1.54 pp, +1.63 pp, +1.08 pp), though none of the 6-fold paired differences remain statistically significant after Holm correction (*p*_Holm >= 0.2251).
   - `WavLM Base+ Full FT` exhibits higher variance across folds (63.06 ± 12.52% clean UAR, including `1 / 18` optimization collapse on Fold 2 at `lr=5e-5`), but degrades more gradually at 0 dB SNR (37.93% white, 46.33% babble).
5. **Cross-corpus generalization gap:**
   - Zero-shot transfer of `wav2vec2 Full FT` from RAVDESS to CREMA-D (6 shared emotions, 3 seeds) yields **42.58 ± 2.27% UAR** (**41.34 ± 2.76% Macro-F1**), highlighting substantial domain shift across corpora.

---

## Methodological & Reproducibility Notes

- **Nested 6-Fold Speaker-Independent Protocol (`splits/folds.json`):**
  - 24 RAVDESS actors are partitioned into 6 speaker-disjoint folds of 4 actors each (`2 male, 2 female`).
  - For each outer fold *k*, fold *k* is held out strictly for testing, fold *(k + 1) mod 6* is used as the validation fold (for per-fold learning-rate selection and early stopping), and the remaining 4 folds (16 actors) form the training set.
- **Per-Fold Learning Rate Selection (`results/lr_selection.json`):**
  - Learning rates `{5e-6, 1e-5, 3e-5, 5e-5, 1e-4}` are swept on seed 42 using each fold's validation UAR/Macro-F1, and the selected per-fold LR is applied across seeds `{42, 43, 44}`.
- **Statistical Aggregation (`src/stats.py`):**
  - Multi-seed runs (`seeds = [42, 43, 44]`) are first averaged **within each fold**, and mean ± std (`ddof=1`) as well as paired *t*-tests, 95% *t*-confidence intervals (*df* = 5), and Wilcoxon signed-rank tests are computed across the **6 fold-level averages**.
  - Noise comparisons across 3 SNRs (`20 dB`, `10 dB`, `0 dB`) are adjusted using Holm-Bonferroni correction.
- **Waveform Normalization Note (`WavLM Base+` vs. `wav2vec2-base`):**
  - While `microsoft/wavlm-base-plus` sets `do_normalize: false` in its official HuggingFace `Wav2Vec2FeatureExtractor` config and `facebook/wav2vec2-base` sets `do_normalize: true`, our pipeline applies per-utterance zero-mean, unit-variance waveform normalization uniformly to both backbones (`normalize_waveform: true` in `src/data.py` and `configs/*.yaml`) so that input scaling is held constant across models.
- **Fixed-Seed Additive Noise (`src/noise.py`):**
  - Every test utterance receives deterministic white Gaussian noise or multi-speaker RAVDESS babble noise seeded by a stable hash of `(utterance_id, snr_db, noise_type)`, ensuring all compared models are evaluated on identical noisy waveforms.

---

## Summary of Results

All tables below are directly reproducible from `results/predictions/` via `python scripts/make_tables.py` (saved to [tables_III_to_VI.md](results/tables/tables_III_to_VI.md)).

### Main Results & Inference Efficiency (RAVDESS, 6-Fold Nested CV)

| Model | Seeds | UAR (%) | WA (%) | Macro-F1 (%) | Trainable Params (M) | CPU Median (ms/utt) | GPU Median (ms/utt) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D-CNN (Mel) | 1 | 44.60 ± 9.72 | 45.00 ± 9.08 | 42.26 ± 11.26 | 0.10 | 2.58 | 1.04 |
| Bi-LSTM (MFCC) | 1 | 39.84 ± 3.80 | 40.97 ± 3.01 | 38.05 ± 3.55 | 0.57 | 33.57 | 4.89 |
| wav2vec2 Linear Probe | 1 | 42.19 ± 5.50 | 42.15 ± 4.69 | 40.11 ± 5.21 | 0.01 | 250.95 | 11.57 |
| wav2vec2 Weighted Sum | 1 | 56.64 ± 5.90 | 55.97 ± 4.55 | 53.69 ± 5.55 | 0.01 | 263.81 | 11.65 |
| wav2vec2 Partial FT (L8-L11) | 3 | 72.37 ± 6.89 | 72.31 ± 6.70 | 71.10 ± 6.76 | 90.18 | 264.74 | 11.53 |
| wav2vec2 Full FT | 3 | 71.55 ± 6.11 | 71.57 ± 6.46 | 70.30 ± 6.49 | 94.38 | 270.06 | 11.53 |
| WavLM Base+ Full FT | 3 | 63.06 ± 12.52 | 63.24 ± 12.39 | 60.57 ± 15.13 | 94.39 | 294.90 | 18.01 |

### Ablations & Paired Comparisons (UAR, *df* = 5)

| Comparison | Mean Diff (pp) | 95% *t*-CI (pp) | Paired *t* (*p*) | Wilcoxon (*p*) |
|---|---:|---:|---:|---:|
| `wav2vec2 Full FT` vs. `Partial FT (L8-L11)` | -0.83 | [-4.40, +2.75] | -0.59 (0.5793) | 0.6875 |
| `wav2vec2 Full FT` vs. `Center Crop` | -0.37 | [-5.31, +4.57] | -0.19 (0.8556) | 1.0000 |
| `wav2vec2 Full FT` vs. `WavLM Base+ Full FT` | +8.49 | [-9.43, +26.40] | +1.22 (0.2776) | 0.4375 |
| `wav2vec2 Full FT` vs. `Weighted Sum` | +14.91 | [+3.31, +26.51] | +3.30 (0.0214) | 0.0312 |
| `wav2vec2 Full FT` vs. `Unweighted CE` | -0.02 | [-4.22, +4.18] | -0.01 (0.9899) | 0.6250 |

### Noise Robustness (3 Seeds, UAR %)

| Noise Type | Model | Clean | 20 dB | 10 dB | 0 dB |
|---|---|---:|---:|---:|---:|
| White | wav2vec2 Full FT | 71.55 ± 6.11 | 64.26 ± 8.72 | 58.46 ± 10.40 | 32.47 ± 11.26 |
| White | wav2vec2 Partial FT | 72.37 ± 6.89 | 60.05 ± 7.29 | 52.21 ± 5.52 | 24.96 ± 6.73 |
| White | WavLM Base+ Full FT | 63.06 ± 12.52 | 56.36 ± 11.74 | 51.52 ± 11.36 | 37.93 ± 9.61 |
| Babble | wav2vec2 Full FT | 71.55 ± 6.11 | 63.17 ± 7.35 | 53.39 ± 5.97 | 41.78 ± 4.07 |
| Babble | wav2vec2 Partial FT | 72.37 ± 6.89 | 61.63 ± 4.68 | 51.76 ± 4.80 | 40.69 ± 4.36 |
| Babble | WavLM Base+ Full FT | 63.06 ± 12.52 | 59.64 ± 11.05 | 55.10 ± 10.69 | 46.33 ± 7.43 |

---

## Repository Structure

```text
ser-adaptation-depth/
├── README.md
├── LICENSE
├── requirements.txt
├── splits/
│   ├── folds.json                     # 6-fold speaker-independent nested splits (train/val/test actors)
│   └── ravdess_file_list.txt          # 1,440 deduplicated RAVDESS speech filenames
├── configs/                           # Experiment YAML configurations
│   ├── w2v2_full_ft.yaml
│   ├── w2v2_partial_ft.yaml
│   ├── w2v2_linear_probe.yaml
│   ├── w2v2_weighted_sum.yaml
│   ├── wavlm_full_ft.yaml
│   ├── ablation_center_crop.yaml
│   ├── ablation_unweighted_ce.yaml
│   ├── cnn_mel.yaml
│   └── bilstm_mfcc.yaml
├── src/
│   ├── data.py                        # Audio loading, resampling, normalization, crop/pad
│   ├── models.py                      # SSLClassifier, 2D-CNN, Bi-LSTM
│   ├── train.py                       # Nested 6-fold training + validation early stopping
│   ├── lr_selection.py                # Per-fold learning-rate selection
│   ├── noise.py                       # Fixed-seed white and babble noise injection
│   ├── evaluate.py                    # Clean, noisy, and cross-corpus CREMA-D evaluation
│   ├── stats.py                       # Paired t-test, Wilcoxon, Holm correction, 95% CI
│   └── latency.py                     # CPU/GPU inference latency benchmarking
├── scripts/
│   ├── run_all.sh                     # End-to-end reproduction pipeline
│   ├── make_tables.py                 # Generates Tables III-VI from predictions & summaries
│   └── make_figures.py                # Generates Fig. 2 (confusion matrices) & Fig. 3 (noise curves)
└── results/
    ├── predictions/                   # Per-model x per-fold x per-seed prediction JSONs
    ├── tables/                        # Generated Markdown & JSON tables
    ├── figures/                       # Generated PNG figures
    ├── lr_selection.json
    ├── master_analysis_report.json
    ├── crema_cross_corpus_summary.json
    ├── efficiency_and_lengths.json
    └── *_robustness_3seeds_summary.json
```

---

## Setup & Reproduction

### 1. Environment

```bash
conda create -n ser-depth python=3.10 -y
conda activate ser-depth
pip install -r requirements.txt
```

### 2. Datasets

Audio files are not distributed in this repository. Download:
- **RAVDESS** (Speech audio-only, 24 actors, 1,440 files matching `splits/ravdess_file_list.txt`)
- **CREMA-D** (`AudioWAV/` directory for zero-shot cross-corpus evaluation)

### 3. Regenerate Tables & Figures from Included Predictions

You can immediately reproduce all paper tables and figures from the included `results/predictions/` without downloading the audio datasets or retraining:

```bash
python scripts/make_tables.py
python scripts/make_figures.py
```

### 4. Run the Full Training & Evaluation Pipeline

```bash
RAVDESS_DIR=/path/to/ravdess CREMAD_DIR=/path/to/cremad bash scripts/run_all.sh
```

---

## License

This project is released under the [MIT License](LICENSE).
