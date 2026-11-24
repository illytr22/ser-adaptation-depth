#!/usr/bin/env bash
# End-to-end reproduction script for nested 6-fold SER adaptation-depth experiments.
# Usage:
#   RAVDESS_DIR=/path/to/ravdess CREMAD_DIR=/path/to/cremad bash scripts/run_all.sh

set -euo pipefail

RAVDESS_DIR="${RAVDESS_DIR:-./data/ravdess}"
CREMAD_DIR="${CREMAD_DIR:-./data/cremad}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== Step 1: Nested per-fold learning rate selection (seed 42) ==="
"$PYTHON_BIN" -m src.lr_selection \
  --ravdess_dir "$RAVDESS_DIR" \
  --splits splits/folds.json \
  --output results/lr_selection.json

echo "=== Step 2: Train all models across 6 folds ==="
CONFIGS_3_SEEDS=(
  "configs/w2v2_full_ft.yaml"
  "configs/w2v2_partial_ft.yaml"
  "configs/wavlm_full_ft.yaml"
  "configs/ablation_center_crop.yaml"
  "configs/ablation_unweighted_ce.yaml"
)

CONFIGS_1_SEED=(
  "configs/w2v2_linear_probe.yaml"
  "configs/w2v2_weighted_sum.yaml"
  "configs/cnn_mel.yaml"
  "configs/bilstm_mfcc.yaml"
)

for cfg in "${CONFIGS_3_SEEDS[@]}"; do
  for seed in 42 43 44; do
    echo "Training $cfg (seed=$seed)..."
    "$PYTHON_BIN" -m src.train \
      --config "$cfg" \
      --ravdess_dir "$RAVDESS_DIR" \
      --splits splits/folds.json \
      --lr_selection results/lr_selection.json \
      --seed "$seed"
  done
done

for cfg in "${CONFIGS_1_SEED[@]}"; do
  echo "Training $cfg (seed=42)..."
  "$PYTHON_BIN" -m src.train \
    --config "$cfg" \
    --ravdess_dir "$RAVDESS_DIR" \
    --splits splits/folds.json \
    --seed 42
done

echo "=== Step 3: Noise robustness & cross-corpus CREMA-D evaluation ==="
"$PYTHON_BIN" -m src.evaluate \
  --ravdess_dir "$RAVDESS_DIR" \
  --cremad_dir "$CREMAD_DIR" \
  --splits splits/folds.json \
  --preds_dir results/predictions \
  --output_dir results

echo "=== Step 4: Inference latency benchmark ==="
"$PYTHON_BIN" -m src.latency \
  --output results/efficiency_and_lengths.json

echo "=== Step 5: Generate Tables III-VI and Figures 2-3 ==="
"$PYTHON_BIN" scripts/make_tables.py
"$PYTHON_BIN" scripts/make_figures.py

echo "=== Pipeline complete. Outputs saved under results/ ==="
