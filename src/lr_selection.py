"""Per-fold nested learning rate selection using inner validation Macro-F1."""

import os
import json
import argparse
import yaml
import numpy as np

from .data import load_folds, load_ravdess_data, read_audio_mono_16k
from .train import run_single_fold


def select_lr_per_fold(cfg, all_data, waveform_cache, folds_dict, search_out_root, seed=42):
    """Run all candidate LRs on seed=42 and pick per-fold optimal LR purely on inner validation Macro-F1."""
    exp_name = cfg["experiment_name"]
    candidates = cfg.get("lr_candidates", [5e-6, 1e-5, 3e-5, 5e-5, 1e-4])
    results = {}

    for k in range(1, 7):
        val_f1s, test_uars = {}, {}
        for lr in candidates:
            lr_tag = f"{lr:.0e}".replace("-0", "-")
            out_dir = os.path.join(search_out_root, f"{exp_name}_lr_{lr_tag}")
            log = run_single_fold(cfg, all_data, waveform_cache, folds_dict, k, seed, lr, out_dir)
            val_f1s[lr_tag] = log["best_val_macro_f1"]
            test_uars[lr_tag] = log["test_metrics"]["uar"]
        best_tag = max(val_f1s, key=lambda t: val_f1s[t])
        results[f"fold_{k}"] = {
            "val_macro_f1": val_f1s,
            "test_uar": test_uars,
            "selected_lr": best_tag,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Run per-fold nested LR selection.")
    parser.add_argument("--configs", nargs="+", default=["configs/w2v2_full_ft.yaml", "configs/wavlm_full_ft.yaml"])
    parser.add_argument("--data_root", type=str, default="data/ravdess_wav")
    parser.add_argument("--folds_path", type=str, default="splits/folds.json")
    parser.add_argument("--search_out_root", type=str, default="results/lr_search")
    parser.add_argument("--out_json", type=str, default="results/lr_selection.json")
    args = parser.parse_args()

    all_data = load_ravdess_data(args.data_root)
    folds_dict = load_folds(args.folds_path)
    waveform_cache = {d["path"]: read_audio_mono_16k(d["path"]) for d in all_data}

    summary = {}
    for cfg_path in args.configs:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        summary[cfg["experiment_name"]] = select_lr_per_fold(cfg, all_data, waveform_cache, folds_dict, args.search_out_root)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved nested LR selection summary to {args.out_json}")


if __name__ == "__main__":
    main()
