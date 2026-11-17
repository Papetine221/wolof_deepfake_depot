#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
equalize_channel.py — Égalisation DSP rapide et précise (bona fide & spoof).

Supprime les raccourcis de canal mesurés par audit_base.py :
  - Durée identique au sein de chaque trio (crop énergie au min du trio, sans zero-padding)
  - Égalisation LTAS (enveloppe spectrale moyenne Welch 1024) avec gabarit commun
  - Plancher de bruit / SNR de trame (P90 - P10) aligné + dither large bande contrôlé
  - Normalisation RMS et crête sans clipping
"""
import glob
import os
import re
import zlib
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm

SR = 16000
N_FFT = 1024
ATTACK_DIRS = {
    "S01": "S01_xtts_wolof",
    "S02": "S02_xtts_multi_ref",
    "S03": "S03_speecht5_wolof",
}


def read_wav_mono(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def energy_crop_np(x, target_len):
    n = len(x)
    if n <= target_len:
        return x
    hop = max(1, SR // 100)
    cs = np.concatenate([[0.0], np.cumsum(x ** 2)])
    starts = np.arange(0, n - target_len + 1, hop)
    energies = cs[starts + target_len] - cs[starts]
    idx = int(starts[int(np.argmax(energies))])
    return x[idx : idx + target_len]


def frame_energy_db(x, sr=SR, win=0.025, hop=0.010):
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def smooth_curve(v, width=5):
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    pad = width // 2
    return np.convolve(np.pad(v, (pad, pad), mode="edge"), kernel, mode="valid")


def compute_ltas_template(paths, sos_bp):
    specs = []
    for p in paths:
        x, _ = read_wav_mono(p)
        x = signal.sosfiltfilt(sos_bp, x)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
        _, pxx = signal.welch(x, fs=SR, nperseg=N_FFT)
        specs.append(np.sqrt(pxx + 1e-14))
    return smooth_curve(np.mean(specs, axis=0), width=7)


def apply_ltas_match(x, target_mag, rng, jitter_std=0.04):
    """Aligne l'enveloppe spectrale LTAS (Welch 1024) tout en gardant la phase et les modulations."""
    _, pxx = signal.welch(x, fs=SR, nperseg=N_FFT)
    cur_mag = smooth_curve(np.sqrt(pxx + 1e-14), width=5)
    # Légère variation aléatoire douce commune à toutes les classes pour éviter un spectre figé
    tilt = np.linspace(-1.0, 1.0, len(target_mag)) * rng.normal(0.0, jitter_std)
    desired = target_mag * (10.0 ** tilt)
    gain = np.clip(desired / (cur_mag + 1e-12), 0.1, 10.0)

    _, _, Zxx = signal.stft(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT * 3 // 4)
    Zxx_eq = Zxx * gain[:, None]
    _, x_eq = signal.istft(Zxx_eq, fs=SR, nperseg=N_FFT, noverlap=N_FFT * 3 // 4)
    return x_eq[: len(x)]


def process_one(x, target_mag, sos_bp, seed_int):
    rng = np.random.RandomState(seed_int)
    x = signal.sosfiltfilt(sos_bp, x)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)

    # 1. Aligner l'enveloppe LTAS de la parole
    x = apply_ltas_match(x, target_mag, rng)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)

    # 2. Aligner la dynamique de trame (SNR P90 - P10 et silence début/fin) via bruit de canal
    w_band = signal.sosfiltfilt(sos_bp, rng.randn(len(x) + 400))[: len(x)]
    w_band = apply_ltas_match(w_band, target_mag, rng, jitter_std=0.01)
    w_band = w_band / (np.sqrt(np.mean(w_band ** 2)) + 1e-9)

    # 3 itérations rapides suffisent pour caler P90 - P10 autour de 17.5 dB
    goal_snr = 17.5 + rng.normal(0.0, 0.9)
    lo, hi = 0.0, 0.40
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        e = frame_energy_db(x + mid * w_band)
        snr = float(np.percentile(e, 90) - np.percentile(e, 10))
        if snr > goal_snr:
            lo = mid
        else:
            hi = mid
    x = x + 0.5 * (lo + hi) * w_band

    # 3. Passe finale LTAS + dither large bande contrôlé (stabilise hf_4k/hf_6k et platitude_spectrale)
    x = apply_ltas_match(x, target_mag, rng, jitter_std=0.03)
    dither_db = -52.0 + rng.normal(0.0, 0.8)
    dither = rng.randn(len(x)) * (10.0 ** (dither_db / 20.0))
    x = x + dither

    # 4. Suppression DC et normalisation RMS (-20 dBFS +/- 0.5 dB)
    x = x - np.mean(x)
    rms_goal = 10.0 ** ((-20.0 + rng.normal(0.0, 0.5)) / 20.0)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * rms_goal

    # Normalisation de crête douce pour aligner peak_db sans clipping
    peak_goal = 10.0 ** ((-3.2 + rng.normal(0.0, 0.3)) / 20.0)
    cur_peak = np.max(np.abs(x)) + 1e-9
    if cur_peak > peak_goal:
        x = peak_goal * np.tanh(x / peak_goal)
    return x


def main():
    in_bona = "data/bonafide_new"
    in_spoof = "data/spoof_new"
    out_root = "data/dataset_equalized"
    out_bona = os.path.join(out_root, "bonafide_new")
    out_spoof = os.path.join(out_root, "spoof_new")

    os.makedirs(out_bona, exist_ok=True)
    for sub in ATTACK_DIRS.values():
        os.makedirs(os.path.join(out_spoof, sub), exist_ok=True)

    bona_files = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(in_bona, "*.wav"))
    }
    spoof_files = defaultdict(dict)
    for tag, sub in ATTACK_DIRS.items():
        for p in glob.glob(os.path.join(in_spoof, sub, "*.wav")):
            stem = re.sub(r"^S0\d_", "", os.path.splitext(os.path.basename(p))[0])
            spoof_files[stem][tag] = p

    sos_bp = signal.butter(4, [130.0, 3400.0], btype="bandpass", fs=SR, output="sos")

    rng_global = np.random.RandomState(42)
    sample_bona = list(bona_files.values())
    sample_spoof = [p for d in spoof_files.values() for p in d.values()]
    rng_global.shuffle(sample_bona)
    rng_global.shuffle(sample_spoof)
    template_paths = sample_bona[:80] + sample_spoof[:80]
    print(f"[INFO] Calcul du gabarit spectral LTAS (Welch 1024) sur {len(template_paths)} fichiers...")
    target_mag = compute_ltas_template(template_paths, sos_bp)

    utts = sorted(bona_files.keys())
    print(f"[INFO] Égalisation de {len(utts)} énoncés vers {out_root}...")
    for utt in tqdm(utts, desc="Égalisation DSP v2"):
        group_paths = [("BONA", bona_files[utt], os.path.join(out_bona, f"{utt}.wav"))]
        for tag, sub in ATTACK_DIRS.items():
            if tag in spoof_files[utt]:
                group_paths.append(
                    (
                        tag,
                        spoof_files[utt][tag],
                        os.path.join(out_spoof, sub, f"{tag}_{utt}.wav"),
                    )
                )

        waves = []
        for tag, src, dst in group_paths:
            x, _ = read_wav_mono(src)
            waves.append((tag, x, dst))

        min_len = min(len(x) for _, x, _ in waves)
        base_seed = zlib.crc32(utt.encode("utf-8"))

        for k, (tag, x, dst) in enumerate(waves):
            x_crop = energy_crop_np(x, min_len)
            file_seed = (base_seed + k * 9973) & 0xFFFFFFFF
            x_out = process_one(x_crop, target_mag, sos_bp, file_seed)
            sf.write(dst, x_out.astype(np.float32), SR, subtype="PCM_16")

    print(f"[INFO] Terminé -> {out_root}")


if __name__ == "__main__":
    main()
