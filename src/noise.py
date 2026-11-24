"""Deterministic additive white Gaussian noise and 6-speaker CREMA-D babble noise generators."""

import numpy as np
from .data import crop_or_pad, read_audio_mono_16k


def add_white_noise(waveform, snr_db, rng):
    """Add zero-mean white Gaussian noise at a specified SNR (dB)."""
    if snr_db is None:
        return waveform
    sig_power = np.mean(waveform ** 2)
    if sig_power < 1e-12:
        return waveform
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape)
    return (waveform + noise).astype(np.float32)


def build_babble_pool(crema_data, num_tracks=30, speakers_per_mix=6, max_len=64000, pool_seed=2026):
    """
    Synthesize multi-talker babble noise tracks by averaging 6 randomly sampled
    CREMA-D utterances per track with a fixed random seed.
    """
    rng = np.random.RandomState(pool_seed)
    indices = rng.choice(len(crema_data), size=min(len(crema_data), num_tracks * speakers_per_mix), replace=False)
    wavs = [crop_or_pad(read_audio_mono_16k(crema_data[idx]["path"]), max_len=max_len, mode="center") for idx in indices]

    babble_pool = []
    for t in range(num_tracks):
        chunk = wavs[t * speakers_per_mix : (t + 1) * speakers_per_mix]
        if not chunk:
            break
        mix = np.mean(chunk, axis=0)
        mix = mix - np.mean(mix)
        std = np.std(mix)
        if std > 1e-8:
            mix = mix / std
        babble_pool.append(mix.astype(np.float32))
    return babble_pool


def add_babble_noise(waveform, snr_db, babble_pool, rng):
    """Superimpose a randomly shifted babble track scaled to target SNR (dB)."""
    if snr_db is None:
        return waveform
    sig_power = np.mean(waveform ** 2)
    if sig_power < 1e-12:
        return waveform
    noise_src = babble_pool[rng.randint(0, len(babble_pool))]
    shift = rng.randint(0, len(noise_src))
    noise_src = np.roll(noise_src, shift)
    noise_power_cur = np.mean(noise_src ** 2)
    if noise_power_cur < 1e-12:
        return waveform
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power_cur)
    return (waveform + scale * noise_src).astype(np.float32)


def get_fold_noise_rng(fold_k, snr_db):
    """
    Return a deterministic RNG keyed only by (fold_k, snr_db), independent of
    model architecture or training seed, ensuring identical noisy test audio across models.
    """
    return np.random.RandomState(202600 + fold_k * 100 + (snr_db if snr_db is not None else 99))
