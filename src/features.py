#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/features.py — Extraction et mise en cache des 4 représentations spectrales (16 kHz) :
  - mfcc : 40 coefficients, n_fft=512, hop=256, n_mels=80
  - lfcc : 60 coefficients, 70 filtres linéaires, n_fft=512, hop=256
  - mel  : 128 bandes, f_max=8000, en dB (n_fft=512, hop=256)
  - cqt  : 84 bins, 12 bins/octave (librosa.cqt, hop=256, en dB)

Mise en cache sous experiments/cache/<condition>/<feature>/<utt_id>.npy (fichier entier,
après normalisation d'amplitude pic).
Standardisation (moyenne/écart-type par bande) calculée sur le train uniquement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy.fft import dct

from src.data import (
    CROP_FRAMES,
    HOP_LENGTH,
    SR,
    ProtocolEntry,
    peak_normalize,
    read_audio_sf,
)

N_FFT = 512
CACHE_ROOT = Path("experiments") / "cache"
VALID_FEATURES = ("mfcc", "lfcc", "mel", "cqt")


def _build_linear_filterbank(
    sr: int = SR, n_fft: int = N_FFT, n_filters: int = 70, f_min: float = 0.0, f_max: float = 8000.0
) -> np.ndarray:
    """Construit une banque de 70 filtres triangulaires linéairement espacés entre 0 et 8000 Hz."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_bins = len(freqs)
    centers = np.linspace(f_min, f_max, n_filters + 2)
    fbank = np.zeros((n_filters, n_bins), dtype=np.float32)

    for m in range(1, n_filters + 1):
        f_left = centers[m - 1]
        f_center = centers[m]
        f_right = centers[m + 1]

        left_mask = (freqs >= f_left) & (freqs <= f_center)
        right_mask = (freqs > f_center) & (freqs <= f_right)

        if f_center > f_left:
            fbank[m - 1, left_mask] = (freqs[left_mask] - f_left) / (f_center - f_left)
        if f_right > f_center:
            fbank[m - 1, right_mask] = (f_right - freqs[right_mask]) / (f_right - f_center)

    return fbank


_LFCC_FBANK_70 = _build_linear_filterbank(sr=SR, n_fft=N_FFT, n_filters=70, f_min=0.0, f_max=8000.0)


def extract_feature_from_waveform(
    audio: np.ndarray, feature_name: str, sr: int = SR
) -> np.ndarray:
    """
    Calcule la représentation spectrale demandée (F, T) à partir d'un signal 1D float32.
    Le signal est préalablement normalisé en amplitude (pic).
    """
    audio = peak_normalize(audio)

    if feature_name == "mfcc":
        # 40 coefficients, n_fft=512, hop=256, n_mels=80
        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=80,
            fmax=8000.0,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        feat = librosa.feature.mfcc(S=mel_db, n_mfcc=40)
        return feat.astype(np.float32)

    if feature_name == "lfcc":
        # 60 coefficients, 70 filtres linéaires, n_fft=512, hop=256
        stft_mag = np.abs(librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        power_spec = stft_mag ** 2
        filter_energies = np.dot(_LFCC_FBANK_70, power_spec)
        log_energies = np.log(np.maximum(filter_energies, 1e-10))
        lfcc = dct(log_energies, type=2, axis=0, norm="ortho")[:60]
        return lfcc.astype(np.float32)

    if feature_name == "mel":
        # 128 bandes, f_max=8000, en dB
        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=128,
            fmax=8000.0,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        return mel_db.astype(np.float32)

    if feature_name == "cqt":
        # 84 bins, 12 bins/octave (librosa.cqt)
        cqt_complex = librosa.cqt(
            y=audio,
            sr=sr,
            hop_length=HOP_LENGTH,
            n_bins=84,
            bins_per_octave=12,
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt_complex), ref=np.max)
        return cqt_db.astype(np.float32)

    raise ValueError(f"Feature inconnue : {feature_name}. Attendu : {VALID_FEATURES}")


def apply_std_floor(std: np.ndarray, mode: str = "relative") -> Tuple[np.ndarray, Dict]:
    """
    Plancher sur l'écart-type par bande (calculé sur le train uniquement).

    mode="relative" (défaut) : floor = max(1e-3 * median(std_bandes), 1e-2) ; std = max(std, floor).
    mode="legacy"            : ancien comportement (plancher absolu 1e-6 uniquement).
    Retourne (std_planchée, infos) ; infos = {std_floor, n_bands_floored, min_std}.
    """
    std = np.asarray(std, dtype=np.float64)
    if mode == "legacy":
        floor = 1e-6
    elif mode == "relative":
        floor = max(1e-3 * float(np.median(std)), 1e-2)
    else:
        raise ValueError(f"std_floor inconnu : {mode}")
    n_floored = int((std < floor).sum())
    out = np.maximum(std, floor).astype(np.float32)
    info = {
        "std_floor_mode": mode,
        "std_floor": float(floor),
        "n_bands_floored": n_floored,
        "n_bands": int(std.size),
        "min_std": float(out.min()),
    }
    return out, info


def extract_and_cache_condition(
    condition: str,
    feature_name: str,
    entries_by_split: Dict[str, List[ProtocolEntry]],
    cache_root: Path = CACHE_ROOT,
    std_floor: str = "relative",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extrait et met en cache toutes les features (train, dev, eval) pour (condition, feature_name)
    sous experiments/cache/<condition>/<feature_name>/<utt_id>.npy.
    Calcule également la moyenne et l'écart-type par bande sur le split 'train' uniquement
    (sur les segments recadrés centrés à 4,0 s pour cohérence statistique) et les sauvegarde
    dans train_stats.npz.

    Retourne (feature_cache_dict, train_mean, train_std).
    """
    feat_cache_dir = cache_root / condition / feature_name
    feat_cache_dir.mkdir(parents=True, exist_ok=True)

    feature_cache: Dict[str, np.ndarray] = {}
    all_entries: List[ProtocolEntry] = []
    for split in ("train", "dev", "eval"):
        all_entries.extend(entries_by_split[split])

    for entry in all_entries:
        npy_path = feat_cache_dir / f"{entry.utt_id}.npy"
        if npy_path.exists():
            feat = np.load(str(npy_path))
        else:
            audio = read_audio_sf(entry.wav_path, target_sr=SR)
            feat = extract_feature_from_waveform(audio, feature_name=feature_name, sr=SR)
            np.save(str(npy_path), feat)
        feature_cache[entry.utt_id] = feat

    stats_path = feat_cache_dir / "train_stats.npz"
    if stats_path.exists():
        stats = np.load(str(stats_path))
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
    else:
        # Calcul sur le train uniquement (sur les trames de 4,0 s centrées de chaque fichier du train)
        train_slices = []
        for entry in entries_by_split["train"]:
            f_mat = feature_cache[entry.utt_id]
            t_len = f_mat.shape[1]
            start = max(0, (t_len - CROP_FRAMES) // 2)
            train_slices.append(f_mat[:, start : start + CROP_FRAMES])
        concat_train = np.concatenate(train_slices, axis=1)  # (F, N_train * CROP_FRAMES)
        mean = concat_train.mean(axis=1).astype(np.float32)
        std = np.maximum(concat_train.std(axis=1), 1e-6).astype(np.float32)
        np.savez(str(stats_path), mean=mean, std=std)

    # Plancher relatif appliqué à la lecture (stats brutes conservées dans train_stats.npz)
    std, _ = apply_std_floor(std, std_floor)
    return feature_cache, mean, std

