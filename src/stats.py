"""Metrics, prediction I/O, collapse detection, paired statistical tests, 95% CIs, and Holm correction."""

import os
import json
from collections import Counter

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def save_preds(save_path, ids, labels, preds, logits, actors=None):
    """Save utterance-level test predictions to JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    records = []
    for i in range(len(labels)):
        item = {
            "id": ids[i] if ids is not None else str(i),
            "label": int(labels[i]),
            "pred": int(preds[i]),
            "logits": [round(float(x), 6) for x in logits[i]],
        }
        if actors is not None:
            item["actor"] = int(actors[i])
        records.append(item)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_preds(json_path):
    """Load utterance-level predictions from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {
        "ids": [r["id"] for r in records],
        "labels": np.array([r["label"] for r in records], dtype=np.int64),
        "preds": np.array([r["pred"] for r in records], dtype=np.int64),
        "logits": np.array([r["logits"] for r in records], dtype=np.float32),
        "actors": np.array([r.get("actor", -1) for r in records], dtype=np.int64),
    }


def compute_metrics(labels, preds, present_labels=None):
    """Compute WA, UAR, Macro-F1, and Weighted-F1 (percentage scale 0-100)."""
    wa = accuracy_score(labels, preds) * 100.0
    kwargs = {"zero_division": 0}
    if present_labels is not None:
        kwargs["labels"] = present_labels
    uar = recall_score(labels, preds, average="macro", **kwargs) * 100.0
    macro_f1 = f1_score(labels, preds, average="macro", **kwargs) * 100.0
    weighted_f1 = f1_score(labels, preds, average="weighted", **kwargs) * 100.0
    return {
        "wa": float(wa),
        "uar": float(uar),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def is_collapsed(preds, threshold=0.5):
    """Return True if a single predicted class accounts for >50% of all predictions."""
    preds = np.asarray(preds)
    if len(preds) == 0:
        return False, -1, 0.0
    cls_id, count = Counter(preds.tolist()).most_common(1)[0]
    ratio = count / float(len(preds))
    return bool(ratio > threshold), int(cls_id), float(ratio)


def aggregate_seeds_and_folds(preds_root, seeds=(42, 43, 44), num_folds=6, present_labels=None):
    """
    Average metrics across seeds within each fold first, then compute
    mean and sample standard deviation (ddof=1) across the 6 folds.
    """
    per_seed_fold = {}
    for s in seeds:
        per_seed_fold[s] = {}
        for f in range(1, num_folds + 1):
            d = load_preds(os.path.join(preds_root, f"seed_{s}", f"fold_{f}", "preds.json"))
            per_seed_fold[s][f] = compute_metrics(d["labels"], d["preds"], present_labels=present_labels)

    summary = {}
    for k in ["wa", "uar", "macro_f1", "weighted_f1"]:
        fold_avgs = [float(np.mean([per_seed_fold[s][f][k] for s in seeds])) for f in range(1, num_folds + 1)]
        summary[k] = {
            "mean": round(float(np.mean(fold_avgs)), 2),
            "std": round(float(np.std(fold_avgs, ddof=1)), 2),
            "fold_values": [round(float(x), 2) for x in fold_avgs],
        }
    return summary, per_seed_fold


def paired_test(fold_vals_a, fold_vals_b, name_a="Model A", name_b="Model B"):
    """
    Compute paired t-test, Wilcoxon signed-rank test, mean difference,
    and exact 95% t-confidence interval (df = n - 1) over 6 paired fold scores.
    """
    a = np.asarray(fold_vals_a, dtype=np.float64)
    b = np.asarray(fold_vals_b, dtype=np.float64)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / float(np.sqrt(len(diff)))
    if se_diff > 0:
        ci_low, ci_high = stats.t.interval(0.95, df=len(diff) - 1, loc=mean_diff, scale=se_diff)
    else:
        ci_low, ci_high = mean_diff, mean_diff

    t_stat, p_val_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_val_w = stats.wilcoxon(a, b)
    except Exception:
        w_stat, p_val_w = 0.0, 1.0

    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_a": round(float(np.mean(a)), 2),
        "std_a": round(float(np.std(a, ddof=1)), 2),
        "mean_b": round(float(np.mean(b)), 2),
        "std_b": round(float(np.std(b, ddof=1)), 2),
        "mean_diff": round(mean_diff, 2),
        "std_diff": round(std_diff, 2),
        "ci95_low": round(float(ci_low), 2),
        "ci95_high": round(float(ci_high), 2),
        "t_stat": round(float(t_stat), 4),
        "p_value_ttest": round(float(p_val_t), 6),
        "w_stat": round(float(w_stat), 4),
        "p_value_wilcoxon": round(float(p_val_w), 6),
        "fold_diffs": [round(float(x), 2) for x in diff],
    }


def holm_correct(p_vals):
    """Apply Holm-Bonferroni step-down multiplicity correction to a list of p-values."""
    n = len(p_vals)
    indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
    adj = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, p * (n - rank))
        running_max = max(running_max, val)
        adj[orig_idx] = round(float(running_max), 6)
    return adj


def pooled_confusion(preds_root, seed=42, seeds_ensemble=None, num_folds=6):
    """Aggregate all 6 outer test folds (N=1,440 utterances) into a pooled 8x8 confusion matrix."""
    all_labels, all_preds = [], []
    for f in range(1, num_folds + 1):
        if seeds_ensemble is not None and len(seeds_ensemble) > 1:
            logits_list = []
            labels_ref = None
            for s in seeds_ensemble:
                d = load_preds(os.path.join(preds_root, f"seed_{s}", f"fold_{f}", "preds.json"))
                logits_list.append(d["logits"])
                labels_ref = d["labels"]
            preds_f = np.argmax(np.mean(logits_list, axis=0), axis=1)
            all_labels.extend(labels_ref.tolist())
            all_preds.extend(preds_f.tolist())
        else:
            d = load_preds(os.path.join(preds_root, f"seed_{seed}", f"fold_{f}", "preds.json"))
            all_labels.extend(d["labels"].tolist())
            all_preds.extend(d["preds"].tolist())

    all_labels = np.asarray(all_labels, dtype=np.int64)
    all_preds = np.asarray(all_preds, dtype=np.int64)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    metrics = compute_metrics(all_labels, all_preds)
    return cm, metrics
