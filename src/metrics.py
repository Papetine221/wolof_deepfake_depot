#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/metrics.py — Métriques d'évaluation pour la détection de deepfakes audio :
  - compute_eer(scores, labels) : spoof = classe positive (1), score = probabilité de spoof.
    Retourne (eer_pct, eer_threshold) avec eer_pct en pourcentage [0, 100].
  - compute_auc(scores, labels) : AUC ROC dans [0, 1].
  - eer_per_system(scores, labels, systems) : EER (%) bona fide contre chaque système S01/S02/S03.
  - bootstrap_ci(scores, labels, n_boot=1000, seed=0) : IC 95 % par bootstrap au niveau énoncé.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> Tuple[float, float]:
    """
    Calcule le taux d'égale erreur (EER en %) et le seuil correspondant.
    Convention : label 1 = spoof (classe positive), label 0 = bonafide (classe négative),
    score = probabilité ou log-vraisemblance d'être spoof.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr, pos_label=1)
    fnr = 1.0 - tpr

    # Recherche du point où |FPR - FNR| est minimal, avec interpolation linéaire
    diff = fpr - fnr
    idx = int(np.nanargmin(np.abs(diff)))

    # Si changement de signe entre idx-1 et idx, interpolation linéaire pour plus de précision
    if idx > 0 and (diff[idx - 1] * diff[idx] <= 0):
        d0, d1 = abs(diff[idx - 1]), abs(diff[idx])
        denom = d0 + d1
        w = d0 / denom if denom > 0 else 0.5
        eer_val = (1.0 - w) * (0.5 * (fpr[idx - 1] + fnr[idx - 1])) + w * (0.5 * (fpr[idx] + fnr[idx]))
        thr = float((1.0 - w) * thresholds[idx - 1] + w * thresholds[idx])
    else:
        eer_val = 0.5 * (fpr[idx] + fnr[idx])
        thr = float(thresholds[idx])

    if not np.isfinite(thr):
        thr = 0.5

    return float(eer_val * 100.0), thr


def compute_auc(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    """Calcule l'aire sous la courbe ROC (AUC dans [0, 1])."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    return float(roc_auc_score(labels_arr, scores_arr))


def eer_per_system(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    systems: Sequence[str],
    attack_list: Tuple[str, ...] = ("S01", "S02", "S03"),
) -> Dict[str, float]:
    """
    Calcule l'EER (%) en opposant l'ensemble des bona fide à chaque système d'attaque
    (S01, S02, S03) pris séparément.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    systems_arr = np.asarray(systems)

    bf_mask = labels_arr == 0
    results: Dict[str, float] = {}

    for atk in attack_list:
        atk_mask = (labels_arr == 1) & (systems_arr == atk)
        if np.sum(bf_mask) > 0 and np.sum(atk_mask) > 0:
            sub_mask = bf_mask | atk_mask
            eer_pct, _ = compute_eer(scores_arr[sub_mask], labels_arr[sub_mask])
            results[atk] = float(eer_pct)
        else:
            results[atk] = float("nan")

    return results


def bootstrap_ci(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """
    Calcule l'intervalle de confiance à 95 % (percentiles 2,5 et 97,5) par bootstrap
    (1 000 rééchantillonnages au niveau énoncé, stratifiés par classe pour garantir la
    présence des deux classes dans chaque tirage).
    Retourne {'eer_ci95': (low_pct, high_pct), 'auc_ci95': (low, high)}.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)

    idx_bf = np.where(labels_arr == 0)[0]
    idx_sp = np.where(labels_arr == 1)[0]
    n_bf, n_sp = len(idx_bf), len(idx_sp)

    rng = np.random.default_rng(seed)
    eers = np.empty(n_boot, dtype=np.float64)
    aucs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        s_bf = rng.choice(idx_bf, size=n_bf, replace=True)
        s_sp = rng.choice(idx_sp, size=n_sp, replace=True)
        boot_idx = np.concatenate([s_bf, s_sp])
        b_scores = scores_arr[boot_idx]
        b_labels = labels_arr[boot_idx]
        eers[b], _ = compute_eer(b_scores, b_labels)
        aucs[b] = compute_auc(b_scores, b_labels)

    eer_low, eer_high = np.percentile(eers, [2.5, 97.5])
    auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])

    return {
        "eer_ci95": (float(eer_low), float(eer_high)),
        "auc_ci95": (float(auc_low), float(auc_high)),
    }

