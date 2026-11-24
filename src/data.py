"""Dataset loading, resampling, normalization, fixed-length cropping/padding, and length statistics."""

import os
import json
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

CREMA_TO_RAVDESS_MAP = {
    "NEU": 0,  # Neutral
    "HAP": 2,  # Happy
    "SAD": 3,  # Sad
    "ANG": 4,  # Angry
    "FEA": 5,  # Fearful
    "DIS": 6,  # Disgust
}
CREMA_MASK_INDICES = [1, 7]  # Calm, Surprised
CREMA_LABELS_PRESENT = [0, 2, 3, 4, 5, 6]
CREMA_TARGET_NAMES = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]


def load_folds(folds_path="splits/folds.json"):
    """Load the 6-fold speaker-independent split definition."""
    with open(folds_path, "r", encoding="utf-8") as f:
        return json.load(f)["folds"]


def load_ravdess_data(root_dir="data/ravdess_wav", target_sr=16000):
    """
    Load deduplicated RAVDESS speech utterances (1,440 clips across 24 actors).
    Audio modality is filtered to speech ('01') and deduplicated by utterance signature.
    """
    data = []
    seen_signatures = set()

    for actor in sorted(os.listdir(root_dir)):
        actor_path = os.path.join(root_dir, actor)
        if not os.path.isdir(actor_path):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if not fname.endswith(".wav"):
                continue
            parts = fname.split("-")
            try:
                if parts[1] != "01":  # Speech only
                    continue
                signature = "-".join(parts[1:]).replace(".wav", "")
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                emotion_id = int(parts[2])
                emotion_name = list(LABEL_MAP.keys())[emotion_id - 1]
                actor_id = int(parts[6].split(".")[0])

                data.append({
                    "id": signature,
                    "path": os.path.join(actor_path, fname),
                    "label": LABEL_MAP[emotion_name],
                    "actor": actor_id,
                })
            except Exception:
                continue

    data.sort(key=lambda x: (x["actor"], x["id"]))
    return data


def make_split(all_data, fold_k, folds_dict=None):
    """
    Nested 6-fold speaker-independent split:
      - Fold k (4 actors, 240 clips): outer test fold
      - Fold (k % 6) + 1 (4 actors, 240 clips): inner validation fold
      - Remaining 16 actors (960 clips): training set
    """
    if folds_dict is not None and str(fold_k) in folds_dict:
        info = folds_dict[str(fold_k)]
        test_actors = set(info["test_actors"])
        val_actors = set(info["val_actors"])
    else:
        val_fold = (fold_k % 6) + 1
        fold_actors = {f: list(range((f - 1) * 4 + 1, f * 4 + 1)) for f in range(1, 7)}
        test_actors = set(fold_actors[fold_k])
        val_actors = set(fold_actors[val_fold])

    train_data = [d for d in all_data if d["actor"] not in test_actors and d["actor"] not in val_actors]
    val_data = [d for d in all_data if d["actor"] in val_actors]
    test_data = [d for d in all_data if d["actor"] in test_actors]
    return train_data, val_data, test_data


def load_crema_data(root_dir="data/CREMA_D/AudioWAV"):
    """Load CREMA-D utterances mapped to the 6 overlapping RAVDESS emotion indices."""
    data = []
    for fname in sorted(os.listdir(root_dir)):
        if not fname.endswith(".wav"):
            continue
        parts = fname.split("_")
        try:
            emotion_code = parts[2]
            if emotion_code in CREMA_TO_RAVDESS_MAP:
                data.append({
                    "id": fname.replace(".wav", ""),
                    "path": os.path.join(root_dir, fname),
                    "label": CREMA_TO_RAVDESS_MAP[emotion_code],
                })
        except Exception:
            continue
    return data


def read_audio_mono_16k(path, target_sr=16000):
    """Read waveform as mono float32 and resample to target_sr if needed."""
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        import torchaudio
        tensor_wav = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(tensor_wav, sr, target_sr).squeeze(0).numpy()
    return wav


def crop_or_pad(waveform, max_len=64000, mode="center", rng=None):
    """
    Crop or right-zero-pad a 1D waveform to fixed length max_len (default 4.0s @ 16 kHz).
    mode='random' for training augmentation; mode='center' for evaluation and ablation.
    """
    if len(waveform) > max_len:
        if mode == "random":
            start = rng.randint(0, len(waveform) - max_len) if rng is not None else random.randint(0, len(waveform) - max_len)
        elif mode == "center":
            start = (len(waveform) - max_len) // 2
        else:
            raise ValueError(f"Unknown crop mode: {mode}")
        waveform = waveform[start : start + max_len]
    elif len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    return waveform.astype(np.float32)


def normalize_waveform_tensor(tensor):
    """Zero-mean, unit-variance normalization per utterance (equivalent to Wav2Vec2FeatureExtractor)."""
    mean = tensor.mean(dim=-1, keepdim=True)
    var = tensor.var(dim=-1, unbiased=False, keepdim=True)
    return (tensor - mean) / torch.sqrt(var + 1e-7)


def length_stats(all_data, threshold_sec=4.0):
    """Compute utterance duration statistics and count clips exceeding threshold_sec."""
    durations = []
    for d in all_data:
        info = sf.info(d["path"])
        durations.append(info.frames / float(info.samplerate))
    durations = np.asarray(durations)
    over_count = int(np.sum(durations > threshold_sec))
    total_count = len(durations)
    return {
        "total_clips": total_count,
        "over_4s_count": over_count,
        "over_4s_pct": round(over_count / total_count * 100.0, 2),
        "min_sec": round(float(np.min(durations)), 4),
        "max_sec": round(float(np.max(durations)), 4),
        "mean_sec": round(float(np.mean(durations)), 4),
        "median_sec": round(float(np.median(durations)), 4),
        "std_sec": round(float(np.std(durations, ddof=1)), 4),
    }


class UnifiedAudioDataset(Dataset):
    """Unified dataset returning fixed-length waveforms with optional per-utterance normalization."""

    def __init__(self, data_list, waveform_cache=None, normalize_ssl=True, train=True, train_crop_mode="random", max_len=64000):
        self.data_list = data_list
        self.waveform_cache = waveform_cache
        self.normalize_ssl = normalize_ssl
        self.train = train
        self.train_crop_mode = train_crop_mode
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if self.waveform_cache is not None and item["path"] in self.waveform_cache:
            wav = self.waveform_cache[item["path"]]
        else:
            wav = read_audio_mono_16k(item["path"])
        crop_mode = self.train_crop_mode if self.train else "center"
        wav = crop_or_pad(wav, max_len=self.max_len, mode=crop_mode)
        return {
            "id": item["id"],
            "waveform": wav,
            "label": item["label"],
            "actor": item["actor"],
        }

    def collate_fn(self, batch):
        ids = [b["id"] for b in batch]
        waveforms = np.stack([b["waveform"] for b in batch], axis=0)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        actors = [b["actor"] for b in batch]

        input_values = torch.from_numpy(waveforms)
        if self.normalize_ssl:
            input_values = normalize_waveform_tensor(input_values)

        return {
            "ids": ids,
            "input_values": input_values,
            "attention_mask": None,
            "labels": labels,
            "actors": actors,
        }
