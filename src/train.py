"""Nested 6-fold cross-validation training with inner-validation Macro-F1 early stopping."""

import os
import gc
import json
import time
import random
import argparse
from collections import Counter

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .data import UnifiedAudioDataset, load_folds, load_ravdess_data, make_split, read_audio_mono_16k
from .models import build_model_from_config
from .stats import compute_metrics, is_collapsed, save_preds


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_loader(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    all_ids, all_labels, all_preds, all_logits, all_actors = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs).float()
            preds = torch.argmax(logits, dim=1)
            all_ids.extend(batch["ids"])
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
            all_actors.extend(batch["actors"])
    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_ids, all_labels, all_preds, all_logits, all_actors


def run_single_fold(cfg, all_data, waveform_cache, folds_dict, fold_k, seed, lr, out_dir, ckpt_dir=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_out = os.path.join(out_dir, f"seed_{seed}", f"fold_{fold_k}")
    os.makedirs(fold_out, exist_ok=True)
    pred_path = os.path.join(fold_out, "preds.json")
    log_path = os.path.join(fold_out, "train_log.json")

    if os.path.exists(pred_path) and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)

    set_seed(seed)
    train_data, val_data, test_data = make_split(all_data, fold_k, folds_dict=folds_dict)
    model, normalize_ssl = build_model_from_config(cfg)
    model.to(device)

    train_ds = UnifiedAudioDataset(
        train_data, waveform_cache=waveform_cache, normalize_ssl=normalize_ssl,
        train=True, train_crop_mode=cfg.get("train_crop_mode", "random"),
    )
    val_ds = UnifiedAudioDataset(
        val_data, waveform_cache=waveform_cache, normalize_ssl=normalize_ssl,
        train=False, train_crop_mode="center",
    )
    test_ds = UnifiedAudioDataset(
        test_data, waveform_cache=waveform_cache, normalize_ssl=normalize_ssl,
        train=False, train_crop_mode="center",
    )

    g = torch.Generator()
    g.manual_seed(seed)
    bs = int(cfg.get("batch_size", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=train_ds.collate_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=max(bs, 8), shuffle=False, collate_fn=val_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=max(bs, 8), shuffle=False, collate_fn=test_ds.collate_fn)

    if cfg.get("weighted_loss", True):
        counts = Counter(d["label"] for d in train_data)
        total = len(train_data)
        weights = [total / (8.0 * counts.get(i, 1)) for i in range(8)]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    epochs = int(cfg.get("epochs", 20))
    patience = int(cfg.get("patience", 5))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    use_amp = device.type == "cuda" and cfg["model_type"] in ("wav2vec2", "wavlm")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1, best_epoch, patience_counter = -1.0, 0, 0
    best_state_dict = None
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if cfg["model_type"] in ("wav2vec2", "wavlm") and cfg.get("mode") in ("linear_probe", "weighted_sum"):
            model.backbone.eval()

        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        val_metrics, *_ = evaluate_loader(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / len(train_loader), 4),
            "val_wa": round(val_metrics["wa"], 2),
            "val_uar": round(val_metrics["uar"], 2),
            "val_macro_f1": round(val_f1, 2),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    if ckpt_dir is not None:
        ckpt_fold = os.path.join(ckpt_dir, f"seed_{seed}", f"fold_{fold_k}")
        os.makedirs(ckpt_fold, exist_ok=True)
        torch.save(best_state_dict, os.path.join(ckpt_fold, "best_model.pt"))

    test_metrics, test_ids, test_labels, test_preds, test_logits, test_actors = evaluate_loader(model, test_loader, device)
    save_preds(pred_path, test_ids, test_labels, test_preds, test_logits, actors=test_actors)
    collapsed, max_cls, max_ratio = is_collapsed(test_preds)

    log_data = {
        "seed": seed,
        "fold": fold_k,
        "lr": float(lr),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 2),
        "test_metrics": {k: round(v, 2) for k, v in test_metrics.items()},
        "collapsed": collapsed,
        "dominant_class": max_cls,
        "dominant_ratio": round(max_ratio, 4),
        "elapsed_sec": round(time.time() - t0, 1),
        "history": history,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    del model, optimizer, scheduler, scaler, best_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log_data


def main():
    parser = argparse.ArgumentParser(description="Run nested 6-fold CV experiment from YAML config.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/ravdess_wav")
    parser.add_argument("--folds_path", type=str, default="splits/folds.json")
    parser.add_argument("--out_root", type=str, default="results/predictions")
    parser.add_argument("--ckpt_root", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_data = load_ravdess_data(args.data_root)
    folds_dict = load_folds(args.folds_path)
    waveform_cache = {d["path"]: read_audio_mono_16k(d["path"]) for d in all_data}

    exp_out = os.path.join(args.out_root, cfg["experiment_name"])
    ckpt_out = os.path.join(args.ckpt_root, cfg["experiment_name"]) if args.ckpt_root else None

    for seed in cfg["seeds"]:
        for fold_k in range(1, 7):
            lr = cfg["lr_per_fold"][fold_k] if "lr_per_fold" in cfg else cfg["lr"]
            log = run_single_fold(cfg, all_data, waveform_cache, folds_dict, fold_k, seed, lr, exp_out, ckpt_dir=ckpt_out)
            print(f"[{cfg['experiment_name']}] Seed {seed} Fold {fold_k} (lr={lr}) -> Test UAR={log['test_metrics']['uar']:.2f}%")


if __name__ == "__main__":
    main()
