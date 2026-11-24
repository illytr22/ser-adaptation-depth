"""Unified latency benchmark (batch=1, 4.0s input, 20 warmup, 100 timed runs, median, CPU threads=4)."""

import os
import gc
import json
import time
import platform
import argparse
import yaml
import torch

from .data import length_stats, load_ravdess_data
from .models import build_model_from_config, count_params


def get_cpu_name():
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if name:
                return name.strip()
        except Exception:
            pass
    return platform.processor()


def benchmark(model, duration_sec=4.0, sr=16000, warmup=20, runs=100, threads=4):
    torch.set_num_threads(threads)
    model.eval()
    dummy = torch.randn(1, int(sr * duration_sec), dtype=torch.float32)

    res = {
        "threads": threads,
        "cpu_model": get_cpu_name(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": warmup,
        "runs": runs,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        m_gpu = model.to("cuda")
        x_gpu = dummy.to("cuda")
        with torch.no_grad():
            for _ in range(warmup):
                _ = m_gpu(x_gpu)
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        gpu_times = []
        with torch.no_grad():
            for _ in range(runs):
                starter.record()
                _ = m_gpu(x_gpu)
                ender.record()
                torch.cuda.synchronize()
                gpu_times.append(starter.elapsed_time(ender))
        res["gpu_median_ms"] = round(float(torch.tensor(gpu_times).median().item()), 2)
        res["gpu_rtf"] = round(res["gpu_median_ms"] / (duration_sec * 1000.0), 5)
        del x_gpu

    m_cpu = model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for _ in range(warmup):
            _ = m_cpu(dummy)
    cpu_times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = m_cpu(dummy)
            cpu_times.append((time.perf_counter() - t0) * 1000.0)
    res["cpu_median_ms"] = round(float(torch.tensor(cpu_times).median().item()), 2)
    res["cpu_rtf"] = round(res["cpu_median_ms"] / (duration_sec * 1000.0), 5)
    return res


def main():
    parser = argparse.ArgumentParser(description="Benchmark model parameters, latency, and RAVDESS duration stats.")
    parser.add_argument("--data_root", type=str, default="data/ravdess_wav")
    parser.add_argument("--out_json", type=str, default="results/efficiency_and_lengths.json")
    args = parser.parse_args()

    configs = [
        ("2D-CNN (Mel)", "configs/cnn_mel.yaml"),
        ("Bi-LSTM (MFCC)", "configs/bilstm_mfcc.yaml"),
        ("wav2vec2 linear_probe", "configs/w2v2_linear_probe.yaml"),
        ("wav2vec2 weighted_sum", "configs/w2v2_weighted_sum.yaml"),
        ("wav2vec2 partial_ft", "configs/w2v2_partial_ft.yaml"),
        ("wav2vec2 full_ft", "configs/w2v2_full_ft.yaml"),
        ("WavLM Base+ full_ft", "configs/wavlm_full_ft.yaml"),
    ]

    out = {"length_stats": length_stats(load_ravdess_data(args.data_root)), "benchmarks": {}}
    for label, cfg_path in configs:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model, _ = build_model_from_config(cfg)
        out["benchmarks"][label] = {**count_params(model), **benchmark(model)}
        del model
        gc.collect()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved efficiency and length stats to {args.out_json}")


if __name__ == "__main__":
    main()
