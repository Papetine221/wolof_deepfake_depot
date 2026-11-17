#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/gmm_baseline.py — Baseline GMM-LFCC :
  - 2 GMM de 32 composantes à covariance diagonale (un pour bonafide, un pour spoof),
    entraînés sur les trames LFCC standardisées du split train (recadrées à 4,0 s).
  - Score d'un énoncé = rapport de log-vraisemblance moyen sur les trames :
    LLR(X) = mean_t [ log p(x_t | GMM_spoof) - log p(x_t | GMM_bonafide) ].
"""
from __future__ import annotations

import csv
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from src.data import CROP_FRAMES, ProtocolEntry, load_all_splits
from src.features import extract_and_cache_condition
from src.metrics import bootstrap_ci, compute_auc, compute_eer, eer_per_system


def _get_standardized_cropped_frames(
    entry: ProtocolEntry,
    feature_cache: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    crop_frames: int = CROP_FRAMES,
) -> np.ndarray:
    """Retourne la matrice (T_crop, F) des trames LFCC centrées et standardisées."""
    feat = feature_cache[entry.utt_id]  # (F, T_full)
    t_len = feat.shape[1]
    start = max(0, (t_len - crop_frames) // 2)
    cropped = feat[:, start : start + crop_frames]
    normed = (cropped - mean[:, None]) / np.maximum(std[:, None], 1e-6)
    return normed.T.astype(np.float64)  # (T_crop, F) pour scikit-learn


def score_utterances_gmm(
    entries: List[ProtocolEntry],
    feature_cache: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    gmm_bonafide: GaussianMixture,
    gmm_spoof: GaussianMixture,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Calcule le score LLR moyen sur les trames pour chaque énoncé."""
    scores = []
    labels = []
    systems = []
    utt_ids = []

    for entry in entries:
        frames = _get_standardized_cropped_frames(entry, feature_cache, mean, std)
        ll_spoof = gmm_spoof.score_samples(frames)
        ll_bf = gmm_bonafide.score_samples(frames)
        llr = float(np.mean(ll_spoof - ll_bf))
        scores.append(llr)
        labels.append(entry.label)
        systems.append(entry.system)
        utt_ids.append(entry.utt_id)

    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32), systems, utt_ids


def _save_scores_csv(
    csv_path: Path,
    utt_ids: List[str],
    systems: List[str],
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "system", "label", "score"])
        for u, s, l, sc in zip(utt_ids, systems, labels, scores):
            label_str = "spoof" if int(l) == 1 else "bonafide"
            writer.writerow([u, s, label_str, f"{float(sc):.6f}"])


def run_gmm_baseline(
    condition: str,
    n_components: int = 32,
    seed: int = 0,
    exp_root: Path = Path("experiments"),
    force: bool = False,
) -> Dict:
    """
    Entraîne et évalue la baseline GMM-LFCC pour une condition donnée.
    Sauvegarde dans experiments/<condition>/gmm_lfcc/seed0/ :
      - gmm_models.pkl
      - scores_dev.csv, scores_eval.csv
      - metrics.json
    """
    out_dir = exp_root / condition / "gmm_lfcc" / f"seed{seed}"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    entries_by_split = load_all_splits(condition)
    feature_cache, mean, std = extract_and_cache_condition(
        condition=condition, feature_name="lfcc", entries_by_split=entries_by_split
    )

    bf_frames_list = []
    sp_frames_list = []
    for entry in entries_by_split["train"]:
        frames = _get_standardized_cropped_frames(entry, feature_cache, mean, std)
        if entry.label == 0:
            bf_frames_list.append(frames)
        else:
            sp_frames_list.append(frames)

    X_bf = np.vstack(bf_frames_list)
    X_sp = np.vstack(sp_frames_list)

    gmm_bf = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=200,
        n_init=2,
        random_state=seed,
        reg_covar=1e-5,
    )
    gmm_sp = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=200,
        n_init=2,
        random_state=seed,
        reg_covar=1e-5,
    )

    gmm_bf.fit(X_bf)
    gmm_sp.fit(X_sp)

    dev_scores, dev_labels, dev_sys, dev_utts = score_utterances_gmm(
        entries_by_split["dev"], feature_cache, mean, std, gmm_bf, gmm_sp
    )
    eval_scores, eval_labels, eval_sys, eval_utts = score_utterances_gmm(
        entries_by_split["eval"], feature_cache, mean, std, gmm_bf, gmm_sp
    )

    dev_eer, dev_thr = compute_eer(dev_scores, dev_labels)
    dev_auc = compute_auc(dev_scores, dev_labels)

    eval_eer, eval_thr = compute_eer(eval_scores, eval_labels)
    eval_auc = compute_auc(eval_scores, eval_labels)
    eval_sys_eer = eer_per_system(eval_scores, eval_labels, eval_sys)
    eval_boot = bootstrap_ci(eval_scores, eval_labels, n_boot=1000, seed=seed)

    elapsed = time.time() - t0

    with open(out_dir / "gmm_models.pkl", "wb") as f:
        pickle.dump({"gmm_bonafide": gmm_bf, "gmm_spoof": gmm_sp, "mean": mean, "std": std}, f)

    _save_scores_csv(out_dir / "scores_dev.csv", dev_utts, dev_sys, dev_labels, dev_scores)
    _save_scores_csv(out_dir / "scores_eval.csv", eval_utts, eval_sys, eval_labels, eval_scores)

    metrics = {
        "condition": condition,
        "feature": "gmm_lfcc",
        "model": "GMM-LFCC (32 comp, diag)",
        "seed": seed,
        "dev_eer": round(dev_eer, 4),
        "dev_auc": round(dev_auc, 6),
        "dev_threshold": round(dev_thr, 6),
        "eval_eer": round(eval_eer, 4),
        "eval_auc": round(eval_auc, 6),
        "eval_threshold": round(eval_thr, 6),
        "eer_per_system": {k: round(v, 4) for k, v in eval_sys_eer.items()},
        "bootstrap_ci95": {
            "eer_pct": [round(eval_boot["eer_ci95"][0], 4), round(eval_boot["eer_ci95"][1], 4)],
            "auc": [round(eval_boot["auc_ci95"][0], 6), round(eval_boot["auc_ci95"][1], 6)],
        },
        "train_duration_sec": round(elapsed, 2),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics

