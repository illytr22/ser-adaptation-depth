# Tables III-VI: Nested 6-Fold RAVDESS & Cross-Corpus CREMA-D Results

## Table III: Main Results and Inference Efficiency (RAVDESS, 6-Fold Speaker-Independent Nested CV)

| Model | Seeds | UAR (%) | WA (%) | Macro-F1 (%) | Trainable Params (M) | CPU Median (ms/utt) | GPU Median (ms/utt) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D-CNN (Mel) | 1 | 44.60 +/- 9.72 | 45.00 +/- 9.08 | 42.26 +/- 11.26 | 0.10 | 2.58 | 1.04 |
| Bi-LSTM (MFCC) | 1 | 39.84 +/- 3.80 | 40.97 +/- 3.01 | 38.05 +/- 3.55 | 0.57 | 33.57 | 4.89 |
| wav2vec2 Linear Probe | 1 | 42.19 +/- 5.50 | 42.15 +/- 4.69 | 40.11 +/- 5.21 | 0.01 | 250.95 | 11.57 |
| wav2vec2 Weighted Sum | 1 | 56.64 +/- 5.90 | 55.97 +/- 4.55 | 53.69 +/- 5.55 | 0.01 | 263.81 | 11.65 |
| wav2vec2 Partial FT (L8-L11) | 3 | 72.37 +/- 6.89 | 72.31 +/- 6.70 | 71.10 +/- 6.76 | 90.18 | 264.74 | 11.53 |
| wav2vec2 Full FT | 3 | 71.55 +/- 6.11 | 71.57 +/- 6.46 | 70.30 +/- 6.49 | 94.38 | 270.06 | 11.53 |
| WavLM Base+ Full FT | 3 | 63.06 +/- 12.52 | 63.24 +/- 12.39 | 60.57 +/- 15.13 | 94.39 | 294.90 | 18.01 |

## Table IV: Ablation Study and Paired Statistical Comparisons (UAR, 6 Folds)

### IV-A. Ablation Summary (3 Seeds)

| Variant | UAR (%) | WA (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| wav2vec2 Full FT (Random Crop, Weighted CE) | 71.55 +/- 6.11 | 71.57 +/- 6.46 | 70.30 +/- 6.49 |
| wav2vec2 Full FT (Center Crop) | 71.92 +/- 7.53 | 71.88 +/- 7.47 | 70.74 +/- 7.82 |
| wav2vec2 Full FT (Unweighted CE) | 71.57 +/- 7.02 | 71.16 +/- 7.20 | 70.08 +/- 7.70 |

### IV-B. Paired Fold-Level Tests (UAR, df=5)

| Comparison | Mean Diff (pp) | 95% t-CI (pp) | Paired t (p) | Wilcoxon (p) |
|---|---:|---:|---:|---:|
| wav2vec2 full FT vs wav2vec2 partial FT | -0.83 | [-4.40, +2.75] | -0.59 (0.5793) | 0.6875 |
| wav2vec2 full FT vs Center Crop | -0.37 | [-5.31, +4.57] | -0.19 (0.8556) | 1.0000 |
| wav2vec2 full FT vs WavLM Base+ full FT | +8.49 | [-9.43, +26.40] | +1.22 (0.2776) | 0.4375 |
| wav2vec2 full FT vs wav2vec2 weighted-sum | +14.91 | [+3.31, +26.51] | +3.30 (0.0214) | 0.0312 |
| wav2vec2 full FT (Weighted CE) vs Unweighted CE | -0.02 | [-4.22, +4.18] | -0.01 (0.9899) | 0.6250 |

## Table V: Noise Robustness Under Fixed-Seed White and Babble Noise (3 Seeds, UAR %)

| Noise Type | Model | Clean | 20 dB | 10 dB | 0 dB |
|---|---|---:|---:|---:|---:|
| White | wav2vec2 Full FT | 71.55 +/- 6.11 | 64.26 +/- 8.72 | 58.46 +/- 10.40 | 32.47 +/- 11.26 |
| White | wav2vec2 Partial FT | 72.37 +/- 6.89 | 60.05 +/- 7.29 | 52.21 +/- 5.52 | 24.96 +/- 6.73 |
| White | WavLM Base+ Full FT | 63.06 +/- 12.52 | 56.36 +/- 11.74 | 51.52 +/- 11.36 | 37.93 +/- 9.61 |
| Babble | wav2vec2 Full FT | 71.55 +/- 6.11 | 63.17 +/- 7.35 | 53.39 +/- 5.97 | 41.78 +/- 4.07 |
| Babble | wav2vec2 Partial FT | 72.37 +/- 6.89 | 61.63 +/- 4.68 | 51.76 +/- 4.80 | 40.69 +/- 4.36 |
| Babble | WavLM Base+ Full FT | 63.06 +/- 12.52 | 59.64 +/- 11.05 | 55.10 +/- 10.69 | 46.33 +/- 7.43 |

### V-B. Holm-Corrected Paired Noise Comparisons (UAR across 3 SNRs)

| Noise | Comparison | SNR | Mean Diff (pp) | 95% CI (pp) | Holm p (t-test) | Holm p (Wilcoxon) |
|---|---|---|---:|---:|---:|---:|
| White | wav2vec2 full FT vs wav2vec2 partial FT | 20dB | +4.21 | [-0.91, +9.33] | 0.2251 | 0.3125 |
| White | wav2vec2 full FT vs wav2vec2 partial FT | 10dB | +6.25 | [-3.59, +16.08] | 0.2251 | 0.3125 |
| White | wav2vec2 full FT vs wav2vec2 partial FT | 0dB | +7.51 | [-1.10, +16.12] | 0.2251 | 0.2812 |
| White | wav2vec2 full FT vs WavLM Base+ full FT | 20dB | +7.90 | [-10.81, +26.60] | 0.9818 | 1.0000 |
| White | wav2vec2 full FT vs WavLM Base+ full FT | 10dB | +6.94 | [-11.11, +25.00] | 0.9818 | 1.0000 |
| White | wav2vec2 full FT vs WavLM Base+ full FT | 0dB | -5.47 | [-24.94, +14.00] | 0.9818 | 1.0000 |
| Babble | wav2vec2 full FT vs wav2vec2 partial FT | 20dB | +1.54 | [-2.40, +5.49] | 0.4776 | 0.6250 |
| Babble | wav2vec2 full FT vs wav2vec2 partial FT | 10dB | +1.63 | [-0.31, +3.56] | 0.2494 | 0.3750 |
| Babble | wav2vec2 full FT vs wav2vec2 partial FT | 0dB | +1.08 | [-1.00, +3.17] | 0.4776 | 0.4375 |
| Babble | wav2vec2 full FT vs WavLM Base+ full FT | 20dB | +3.54 | [-13.49, +20.57] | 1.0000 | 0.9375 |
| Babble | wav2vec2 full FT vs WavLM Base+ full FT | 10dB | -1.72 | [-16.85, +13.41] | 1.0000 | 0.9375 |
| Babble | wav2vec2 full FT vs WavLM Base+ full FT | 0dB | -4.56 | [-15.86, +6.74] | 1.0000 | 0.9375 |

## Table VI: Zero-Shot Cross-Corpus Evaluation on CREMA-D (6 Shared Emotions, 3 Seeds)

| Source Model (RAVDESS) | Seeds | UAR (%) | WA (%) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|---:|---:|
| wav2vec2 Full FT | 3 | 42.58 +/- 2.27 | 42.79 +/- 2.32 | 41.34 +/- 2.76 | 41.49 +/- 2.78 |
