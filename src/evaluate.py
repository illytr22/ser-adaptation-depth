"""Evaluation pipeline: RAVDESS test, CREMA-D cross-corpus inference, and white/babble noise robustness."""

import os
import json
import argparse
import numpy as np
import torch
import yaml

from .data import (
    CREMA_LABELS_PRESENT,
    CREMA_MASK_INDICES,
    crop_or_pad,
    load_crema_data,
    load_folds,
    load_ravdess_data,
    make_split,
    normalize_waveform_tensor,
    read_audio_mono_16k,
)
from .models import build_model_from_config
from .noise import add_babble_noise, add_white_noise, build_babble_pool, get_fold_noise_rng
from .stats import compute_metrics, save_preds


def _infer_batches(model, waveforms_list, device, normalize=True, batch_size=32, mask_indices=None):
    preds_all, logits_all = [], []
    with torch.no_grad():
        for i in range(0, len(waveforms_list), batch_size):
            batch = np.stack(waveforms_list[i : i + batch_size], axis=0)
            iv = torch.from_numpy(batch).to(device)
            if normalize:
                iv = normalize_waveform_tensor(iv)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(iv).float()
            if mask_indices is not None:
                masked = logits.clone()
                masked[:, mask_indices] = float("-inf")
                preds = torch.argmax(masked, dim=1)
            else:
                preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy().tolist())
            logits_all.extend(logits.cpu().numpy().tolist())
    return preds_all, logits_all


def evaluate_crema(cfg, ckpt_root, crema_root="data/CREMA_D/AudioWAV", out_json="results/crema_cross_corpus_summary.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crema_data = load_crema_data(crema_root)
    waveforms = [crop_or_pad(read_audio_mono_16k(d["path"]), max_len=64000, mode="center") for d in crema_data]
    labels = [d["label"] for d in crema_data]
    ids = [d["id"] for d in crema_data]

    exp_name = cfg["experiment_name"]
    seeds = cfg["seeds"]
    per_seed_fold = {s: {} for s in seeds}

    for s in seeds:
        for f in range(1, 7):
            ckpt_path = os.path.join(ckpt_root, exp_name, f"seed_{s}", f"fold_{f}", "best_model.pt")
            model, norm_ssl = build_model_from_config(cfg)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.to(device).eval()

            preds, logits = _infer_batches(model, waveforms, device, normalize=norm_ssl, batch_size=64, mask_indices=CREMA_MASK_INDICES)
            per_seed_fold[s][f] = compute_metrics(labels, preds, present_labels=CREMA_LABELS_PRESENT)

    summary = {"seeds": seeds}
    for k in ["uar", "wa", "macro_f1", "weighted_f1"]:
        fold_avgs = [float(np.mean([per_seed_fold[s][f][k] for s in seeds])) for f in range(1, 7)]
        summary[k] = {
            "mean": round(float(np.mean(fold_avgs)), 2),
            "std": round(float(np.std(fold_avgs, ddof=1)), 2),
            "fold_values": [round(float(x), 2) for x in fold_avgs],
        }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def evaluate_noise(cfg, ckpt_root, ravdess_root="data/ravdess_wav", crema_root="data/CREMA_D/AudioWAV", folds_path="splits/folds.json", noise_type="white", snrs=(None, 20, 10, 0), out_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = load_ravdess_data(ravdess_root)
    folds_dict = load_folds(folds_path)
    babble_pool = build_babble_pool(load_crema_data(crema_root)) if noise_type == "babble" else None

    exp_name = cfg["experiment_name"]
    seeds = cfg["seeds"]
    results = {"noise_type": noise_type, "seeds": seeds, "results": {}}

    for snr in snrs:
        tag = "clean" if snr is None else (f"snr_{snr}dB" if noise_type == "white" else f"babble_{snr}dB")
        per_seed_fold = {s: {} for s in seeds}
        for f in range(1, 7):
            _, _, test_data = make_split(all_data, f, folds_dict=folds_dict)
            rng = get_fold_noise_rng(f, snr)
            wavs, labels = [], []
            for item in test_data:
                w = crop_or_pad(read_audio_mono_16k(item["path"]), max_len=64000, mode="center")
                if snr is not None:
                    w = add_white_noise(w, snr, rng) if noise_type == "white" else add_babble_noise(w, snr, babble_pool, rng)
                wavs.append(w)
                labels.append(item["label"])

            for s in seeds:
                ckpt_path = os.path.join(ckpt_root, exp_name, f"seed_{s}", f"fold_{f}", "best_model.pt")
                model, norm_ssl = build_model_from_config(cfg)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                model.to(device).eval()
                preds, _ = _infer_batches(model, wavs, device, normalize=norm_ssl, batch_size=32)
                per_seed_fold[s][f] = compute_metrics(labels, preds)

        snr_dict = {}
        for k in ["uar", "wa", "macro_f1", "weighted_f1"]:
            fold_avgs = [float(np.mean([per_seed_fold[s][f][k] for s in seeds])) for f in range(1, 7)]
            snr_dict[k] = {
                "mean": round(float(np.mean(fold_avgs)), 2),
                "std": round(float(np.std(fold_avgs, ddof=1)), 2),
                "fold_values": [round(float(x), 2) for x in fold_avgs],
            }
        results["results"][tag] = snr_dict

    suffix = "noise_robustness_3seeds_summary.json" if noise_type == "white" else "babble_robustness_3seeds_summary.json"
    out_path = os.path.join(out_dir, f"{exp_name}_{suffix}")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-corpus CREMA-D and noise robustness.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all", choices=["crema", "white", "babble", "all"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.mode in ("crema", "all"):
        evaluate_crema(cfg, args.ckpt_root)
    if args.mode in ("white", "all"):
        evaluate_noise(cfg, args.ckpt_root, noise_type="white")
    if args.mode in ("babble", "all"):
        evaluate_noise(cfg, args.ckpt_root, noise_type="babble")


if __name__ == "__main__":
    main()
