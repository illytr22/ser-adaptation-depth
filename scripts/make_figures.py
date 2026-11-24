"""Generate Figure 2 (pooled confusion matrices) and Figure 3 (noise curves).

Outputs:
- ``results/figures/fig2_confusion_matrices.png``
- ``results/figures/fig3_noise_robustness.png``
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.stats import pooled_confusion  # noqa: E402


RESULTS_DIR = REPO_ROOT / "results"
PREDS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

EMOTION_LABELS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]


def plot_fig2_confusion_matrices() -> Path:
    models = [
        ("wav2vec2 Full FT (3-Seed Ensemble)", "w2v2_full_ft"),
        ("wav2vec2 Partial FT (3-Seed Ensemble)", "w2v2_partial_ft"),
        ("WavLM Base+ Full FT (3-Seed Ensemble)", "wavlm_full_ft"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)

    for ax, (title, pred_dir) in zip(axes, models):
        cm, metrics = pooled_confusion(
            PREDS_DIR / pred_dir,
            seeds_ensemble=(42, 43, 44),
            num_folds=6,
        )
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=100.0)
        ax.set_title(f"{title}\nUAR={metrics['uar']:.2f}%, WA={metrics['wa']:.2f}%", fontsize=11)
        ax.set_xticks(range(len(EMOTION_LABELS)))
        ax.set_yticks(range(len(EMOTION_LABELS)))
        ax.set_xticklabels(EMOTION_LABELS, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(EMOTION_LABELS, fontsize=9)
        ax.set_xlabel("Predicted Emotion", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("True Emotion", fontsize=10)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                val = cm_norm[i, j]
                color = "white" if val > 55.0 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=axes, shrink=0.85, label="Row-Normalized Recall (%)")
    out_path = FIGURES_DIR / "fig2_confusion_matrices.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_fig3_noise_curves() -> Path:
    snr_labels = ["Clean", "20 dB", "10 dB", "0 dB"]
    x = np.arange(len(snr_labels))

    models = [
        ("wav2vec2 Full FT", "w2v2_full_ft", "o", "-"),
        ("wav2vec2 Partial FT", "w2v2_partial_ft", "s", "--"),
        ("WavLM Base+ Full FT", "wavlm_full_ft", "^", ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)

    noise_configs = [
        ("Additive White Noise", "noise", ["clean", "snr_20dB", "snr_10dB", "snr_0dB"]),
        ("Babble Noise (RAVDESS)", "babble", ["clean", "babble_20dB", "babble_10dB", "babble_0dB"]),
    ]

    for ax, (noise_title, suffix, snr_keys) in zip(axes, noise_configs):
        for label, prefix, marker, linestyle in models:
            path = RESULTS_DIR / f"{prefix}_{suffix}_robustness_3seeds_summary.json"
            with open(path, "r", encoding="utf-8") as f:
                res = json.load(f)["results"]
            means = [res[s]["uar"]["mean"] for s in snr_keys]
            stds = [res[s]["uar"]["std"] for s in snr_keys]
            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                capsize=4,
                label=label,
            )

        ax.set_title(noise_title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(snr_labels, fontsize=10)
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 90)

    axes[0].set_ylabel("Unweighted Average Recall (UAR, %)", fontsize=10)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)

    out_path = FIGURES_DIR / "fig3_noise_robustness.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig2 = plot_fig2_confusion_matrices()
    fig3 = plot_fig3_noise_curves()
    print(f"Saved {fig2} and {fig3}")


if __name__ == "__main__":
    main()
