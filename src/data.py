#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/data.py — Indexation des fichiers, lecture des protocoles et Dataset PyTorch.

Règles strictes :
  - Ne modifie jamais data/.
  - Chargement audio exclusivement avec soundfile (sf.read), jamais torchaudio.load.
  - Pas de zero-padding : recadrage commun à 4,0 s (64 000 échantillons à 16 kHz,
    soit 250 trames avec hop_length=256) : aléatoire en train, centré en dev/eval.
  - Normalisation d'amplitude par fichier (pic) après recadrage.
  - Retourne (feature, label, système, utt_id).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16_000
CROP_SECONDS = 4.0
CROP_SAMPLES = int(SR * CROP_SECONDS)  # 64 000 échantillons
HOP_LENGTH = 256
CROP_FRAMES = CROP_SAMPLES // HOP_LENGTH  # 250 trames

DATA_ROOT = Path("data")
PROTOCOL_DIR = DATA_ROOT / "protocols_new"

CONDITION_DIRS: Dict[str, List[Path]] = {
    "original": [DATA_ROOT / "bonafide_new", DATA_ROOT / "spoof_new"],
    "matched": [DATA_ROOT / "matched"],
    "equalized": [DATA_ROOT / "dataset_equalized"],
}


@dataclass(frozen=True)
class ProtocolEntry:
    spk: str
    utt_id: str
    system: str      # "-" pour bonafide, "S01"/"S02"/"S03" pour spoof
    label_str: str   # "bonafide" ou "spoof"
    label: int       # 0 = bonafide, 1 = spoof
    wav_path: Path


def index_condition_wavs(condition: str, data_root: Path = DATA_ROOT) -> Dict[str, Path]:
    """Indexe tous les fichiers .wav d'une condition par nom sans extension (stem)."""
    if condition not in CONDITION_DIRS:
        raise ValueError(f"Condition inconnue : {condition}. Attendu : {list(CONDITION_DIRS)}")

    roots = [
        data_root / p.relative_to(DATA_ROOT) if data_root != DATA_ROOT else p
        for p in CONDITION_DIRS[condition]
    ]
    wav_map: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Dossier introuvable pour la condition '{condition}': {root}")
        for wav_path in sorted(root.rglob("*.wav")):
            wav_map[wav_path.stem] = wav_path
    return wav_map


def read_protocol_entries(
    split: str,
    condition: str,
    data_root: Path = DATA_ROOT,
    wav_map: Optional[Dict[str, Path]] = None,
) -> List[ProtocolEntry]:
    """
    Lit data/protocols_new/wolof_deepfake.{split}.txt et associe chaque ligne à son fichier .wav.
    Cherche d'abord utt_id, sinon {attaque}_{utt_id}.
    Arrête immédiatement si un fichier manque (exigence 100 % retrouvés).
    """
    if split not in ("train", "dev", "eval"):
        raise ValueError(f"Split invalide : {split}")

    if wav_map is None:
        wav_map = index_condition_wavs(condition, data_root=data_root)

    proto_path = (data_root / "protocols_new" / f"wolof_deepfake.{split}.txt")
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocole introuvable : {proto_path}")

    entries: List[ProtocolEntry] = []
    missing: List[Tuple[str, str, str]] = []

    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            spk, utt_id, attack, _, label_str = parts[:5]
            cand1 = utt_id
            cand2 = f"{attack}_{utt_id}"
            wav_path = wav_map.get(cand1) or wav_map.get(cand2)
            if wav_path is None:
                missing.append((utt_id, attack, label_str))
            else:
                label_int = 1 if label_str == "spoof" else 0
                system_str = attack if label_str == "spoof" else "bonafide"
                entries.append(
                    ProtocolEntry(
                        spk=spk,
                        utt_id=utt_id,
                        system=system_str,
                        label_str=label_str,
                        label=label_int,
                        wav_path=wav_path,
                    )
                )

    if missing:
        raise FileNotFoundError(
            f"[{condition}/{split}] {len(missing)} fichiers manquants ! Exemples : {missing[:10]}"
        )
    return entries


def load_all_splits(
    condition: str, data_root: Path = DATA_ROOT
) -> Dict[str, List[ProtocolEntry]]:
    """Charge les 3 splits (train, dev, eval) pour une condition donnée."""
    wav_map = index_condition_wavs(condition, data_root=data_root)
    return {
        split: read_protocol_entries(split, condition, data_root=data_root, wav_map=wav_map)
        for split in ("train", "dev", "eval")
    }


def read_audio_sf(wav_path: Path, target_sr: int = SR) -> np.ndarray:
    """
    Charge un fichier audio mono en float32 avec soundfile.read exclusivement.
    Vérifie que la fréquence d'échantillonnage est 16 kHz.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def crop_and_peak_normalize(
    audio: np.ndarray,
    crop_samples: int = CROP_SAMPLES,
    mode: str = "center",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Recadre l'audio à exactement crop_samples (64 000 échantillons = 4,0 s) SANS zero-padding,
    puis normalise l'amplitude par le pic du segment recadré.
    """
    n = len(audio)
    if n < crop_samples:
        raise ValueError(
            f"Audio trop court ({n} échantillons < {crop_samples}). Zero-padding interdit !"
        )
    if n == crop_samples:
        cropped = audio
    elif mode == "random":
        if rng is None:
            start = int(np.random.randint(0, n - crop_samples + 1))
        else:
            start = int(rng.integers(0, n - crop_samples + 1))
        cropped = audio[start : start + crop_samples]
    else:  # "center"
        start = (n - crop_samples) // 2
        cropped = audio[start : start + crop_samples]

    peak = float(np.max(np.abs(cropped)))
    return (cropped / (peak + 1e-8)).astype(np.float32)


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalisation d'amplitude par fichier (pic)."""
    peak = float(np.max(np.abs(audio)))
    return (audio / (peak + 1e-8)).astype(np.float32)


class WolofAudioDataset(Dataset):
    """
    Dataset PyTorch retournant (feature, label, system, utt_id).
    Utilise les features mises en cache (F, T_full) du fichier entier, puis applique :
      - Recadrage en trames à CROP_FRAMES = 250 (4,0 s avec hop=256) sans zero-padding :
        crop aléatoire à l'entraînement, crop centré à l'évaluation (dev/eval/external).
      - Standardisation (moyenne/écart-type par bande calculés sur le train uniquement).
    """

    def __init__(
        self,
        entries: List[ProtocolEntry],
        feature_cache: Dict[str, np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
        is_train: bool = False,
        crop_frames: int = CROP_FRAMES,
        seed: int = 0,
        clip: float = 0.0,
    ) -> None:
        super().__init__()
        self.clip = float(clip)  # 0 = pas de clipping
        self.entries = entries
        self.feature_cache = feature_cache
        self.mean = mean.astype(np.float32)[:, None]  # (F, 1)
        self.std = np.maximum(std.astype(np.float32), 1e-6)[:, None]  # (F, 1)
        self.is_train = is_train
        self.crop_frames = crop_frames
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        entry = self.entries[idx]
        feat_full = self.feature_cache[entry.utt_id]  # (F, T_full)
        total_frames = feat_full.shape[1]

        if total_frames < self.crop_frames:
            raise ValueError(
                f"Feature trop courte pour {entry.utt_id}: {total_frames} < {self.crop_frames} trames."
            )

        if total_frames == self.crop_frames:
            feat = feat_full
        elif self.is_train:
            max_start = total_frames - self.crop_frames
            start = int(self.rng.integers(0, max_start + 1))
            feat = feat_full[:, start : start + self.crop_frames]
        else:
            start = (total_frames - self.crop_frames) // 2
            feat = feat_full[:, start : start + self.crop_frames]

        # Standardisation par bande (calculée sur le train)
        feat_norm = (feat - self.mean) / self.std
        if self.clip > 0:
            feat_norm = np.clip(feat_norm, -self.clip, self.clip)
        # Tenseur (1, F, T)
        tensor = torch.from_numpy(np.ascontiguousarray(feat_norm, dtype=np.float32)).unsqueeze(0)
        return tensor, int(entry.label), entry.system, entry.utt_id
