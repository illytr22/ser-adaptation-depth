"""Generate publication-ready Tables III-VI in Markdown and JSON formats.

Reads per-utterance predictions under ``results/predictions/`` and summary
reports under ``results/`` to produce:
- Table III: Main nested 6-fold RAVDESS results and inference efficiency
- Table IV: Ablations and paired statistical comparisons (95% CI, t-test, Wilcoxon)
- Table V: Additive white noise & babble noise robustness (Holm-corrected p-values)
- Table VI: Zero-shot cross-corpus evaluation on CREMA-D (6-class)
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stats import aggregate_seeds_and_folds  # noqa: E402


RESULTS_DIR = REPO_ROOT / "results"
PREDS_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"


def fmt_mean_std(stat: dict) -> str:
    return f"{stat['mean']:.2f} +/- {stat['std']:.2f}"


def fmt_ci(stat: dict) -> str:
    return f"[{stat['ci95_low']:+.2f}, {stat['ci95_high']:+.2f}]"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "master_analysis_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(RESULTS_DIR / "efficiency_and_lengths.json", "r", encoding="utf-8") as f:
        eff = json.load(f)
    with open(RESULTS_DIR / "crema_cross_corpus_summary.json", "r", encoding="utf-8") as f:
        crema = json.load(f)

    # Verify prediction files directly via src.stats
    main_specs = [
        ("2D-CNN (Mel)", "cnn_mel", (42,), "2D-CNN (Mel)"),
        ("Bi-LSTM (MFCC)", "bilstm_mfcc", (42,), "Bi-LSTM (MFCC)"),
        ("wav2vec2 Linear Probe", "w2v2_linear_probe", (42,), "wav2vec2 linear_probe"),
        ("wav2vec2 Weighted Sum", "w2v2_weighted_sum", (42,), "wav2vec2 weighted_sum"),
        ("wav2vec2 Partial FT (L8-L11)", "w2v2_partial_ft", (42, 43, 44), "wav2vec2 partial_ft"),
        ("wav2vec2 Full FT", "w2v2_full_ft", (42, 43, 44), "wav2vec2 full_ft"),
        ("WavLM Base+ Full FT", "wavlm_full_ft", (42, 43, 44), "WavLM Base+ full_ft"),
    ]

    table_iii_rows = []
    for label, pred_name, seeds, bench_key in main_specs:
        summary, _ = aggregate_seeds_and_folds(PREDS_DIR / pred_name, seeds=seeds, num_folds=6)
        b = eff["benchmarks"][bench_key]
        table_iii_rows.append(
            {
                "model": label,
                "seeds": len(seeds),
                "uar": summary["uar"],
                "wa": summary["wa"],
                "macro_f1": summary["macro_f1"],
                "trainable_M": b["trainable_M"],
                "cpu_median_ms": b["cpu_median_ms"],
                "gpu_median_ms": b["gpu_median_ms"],
            }
        )

    # Ablations
    ablation_specs = [
        ("wav2vec2 Full FT (Random Crop, Weighted CE)", "w2v2_full_ft", (42, 43, 44)),
        ("wav2vec2 Full FT (Center Crop)", "ablation_center_crop", (42, 43, 44)),
        ("wav2vec2 Full FT (Unweighted CE)", "ablation_unweighted_ce", (42, 43, 44)),
    ]
    ablation_rows = []
    for label, pred_name, seeds in ablation_specs:
        summary, _ = aggregate_seeds_and_folds(PREDS_DIR / pred_name, seeds=seeds, num_folds=6)
        ablation_rows.append(
            {
                "variant": label,
                "uar": summary["uar"],
                "wa": summary["wa"],
                "macro_f1": summary["macro_f1"],
            }
        )

    paired_rows = []
    for key in [
        "full_ft_vs_partial_ft",
        "full_ft_vs_center_crop",
        "full_ft_vs_wavlm",
        "full_ft_vs_weighted_sum",
        "full_ft_vs_unweighted_ce",
    ]:
        entry = report["paired_tests"][key]["uar"]
        paired_rows.append({"key": key, **entry})

    # Noise robustness (Table V)
    noise_models = [
        ("wav2vec2 Full FT", "w2v2_full_ft"),
        ("wav2vec2 Partial FT", "w2v2_partial_ft"),
        ("WavLM Base+ Full FT", "wavlm_full_ft"),
    ]
    noise_table = {}
    for noise_kind, suffix in [("white", "noise"), ("babble", "babble")]:
        noise_table[noise_kind] = {}
        for label, prefix in noise_models:
            path = RESULTS_DIR / f"{prefix}_{suffix}_robustness_3seeds_summary.json"
            with open(path, "r", encoding="utf-8") as f:
                noise_table[noise_kind][label] = json.load(f)["results"]

    # Build Markdown output
    lines = [
        "# Tables III-VI: Nested 6-Fold RAVDESS & Cross-Corpus CREMA-D Results",
        "",
        "## Table III: Main Results and Inference Efficiency (RAVDESS, 6-Fold Speaker-Independent Nested CV)",
        "",
        "| Model | Seeds | UAR (%) | WA (%) | Macro-F1 (%) | Trainable Params (M) | CPU Median (ms/utt) | GPU Median (ms/utt) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_iii_rows:
        lines.append(
            f"| {r['model']} | {r['seeds']} | {fmt_mean_std(r['uar'])} | "
            f"{fmt_mean_std(r['wa'])} | {fmt_mean_std(r['macro_f1'])} | "
            f"{r['trainable_M']:.2f} | {r['cpu_median_ms']:.2f} | {r['gpu_median_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Table IV: Ablation Study and Paired Statistical Comparisons (UAR, 6 Folds)",
            "",
            "### IV-A. Ablation Summary (3 Seeds)",
            "",
            "| Variant | UAR (%) | WA (%) | Macro-F1 (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in ablation_rows:
        lines.append(
            f"| {r['variant']} | {fmt_mean_std(r['uar'])} | "
            f"{fmt_mean_std(r['wa'])} | {fmt_mean_std(r['macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "### IV-B. Paired Fold-Level Tests (UAR, df=5)",
            "",
            "| Comparison | Mean Diff (pp) | 95% t-CI (pp) | Paired t (p) | Wilcoxon (p) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for p in paired_rows:
        lines.append(
            f"| {p['comparison']} | {p['mean_diff']:+.2f} | {fmt_ci(p)} | "
            f"{p['t_stat']:+.2f} ({fmt_p(p['p_value_ttest'])}) | {fmt_p(p['p_value_wilcoxon'])} |"
        )

    lines.extend(
        [
            "",
            "## Table V: Noise Robustness Under Fixed-Seed White and Babble Noise (3 Seeds, UAR %)",
            "",
            "| Noise Type | Model | Clean | 20 dB | 10 dB | 0 dB |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for noise_kind, title, snr_prefix in [("white", "White", "snr_"), ("babble", "Babble", "babble_")]:
        for label, _ in noise_models:
            res = noise_table[noise_kind][label]
            lines.append(
                f"| {title} | {label} | {fmt_mean_std(res['clean']['uar'])} | "
                f"{fmt_mean_std(res[f'{snr_prefix}20dB']['uar'])} | "
                f"{fmt_mean_std(res[f'{snr_prefix}10dB']['uar'])} | "
                f"{fmt_mean_std(res[f'{snr_prefix}0dB']['uar'])} |"
            )

    lines.extend(
        [
            "",
            "### V-B. Holm-Corrected Paired Noise Comparisons (UAR across 3 SNRs)",
            "",
            "| Noise | Comparison | SNR | Mean Diff (pp) | 95% CI (pp) | Holm p (t-test) | Holm p (Wilcoxon) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for noise_kind, snr_prefix in [("white", "snr_"), ("babble", "babble_")]:
        for comp_key in ["full_ft_vs_partial_ft", "full_ft_vs_wavlm"]:
            comp_dict = report["noise_paired_tests_holm"][noise_kind][comp_key]["uar"]
            for snr_db in ["20dB", "10dB", "0dB"]:
                c = comp_dict[f"{snr_prefix}{snr_db}"]
                lines.append(
                    f"| {noise_kind.capitalize()} | {c['comparison']} | {snr_db} | "
                    f"{c['mean_diff']:+.2f} | {fmt_ci(c)} | "
                    f"{fmt_p(c['p_value_ttest_holm'])} | {fmt_p(c['p_value_wilcoxon_holm'])} |"
                )

    lines.extend(
        [
            "",
            "## Table VI: Zero-Shot Cross-Corpus Evaluation on CREMA-D (6 Shared Emotions, 3 Seeds)",
            "",
            "| Source Model (RAVDESS) | Seeds | UAR (%) | WA (%) | Macro-F1 (%) | Weighted-F1 (%) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| wav2vec2 Full FT | {len(crema['seeds'])} | {fmt_mean_std(crema['uar'])} | "
            f"{fmt_mean_std(crema['wa'])} | {fmt_mean_std(crema['macro_f1'])} | "
            f"{fmt_mean_std(crema['weighted_f1'])} |",
            "",
        ]
    )

    md_path = TABLES_DIR / "tables_III_to_VI.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = TABLES_DIR / "tables_III_to_VI.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "table_iii": table_iii_rows,
                "table_iv_ablations": ablation_rows,
                "table_iv_paired": paired_rows,
                "table_v_noise": noise_table,
                "table_v_holm": report["noise_paired_tests_holm"],
                "table_vi_cremad": crema,
            },
            f,
            indent=2,
        )

    print(f"Saved {md_path} and {json_path}")


if __name__ == "__main__":
    main()
