#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_all.py — Entraînement et évaluation LCNN + baseline GMM-LFCC (TÂCHES 2 & 3).

Usage complet (TÂCHE 3) :
    python train_all.py --conditions original matched equalized --features mfcc lfcc mel cqt --seeds 0 1 2

Usage pilote (TÂCHE 2 : 1 condition original × 1 feature lfcc × 1 graine 0 × 3 époques) :
    python train_all.py --pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import ProtocolEntry, WolofAudioDataset, load_all_splits
from src.features import apply_std_floor, extract_and_cache_condition
from src.gmm_baseline import run_gmm_baseline
from src.metrics import bootstrap_ci, compute_auc, compute_eer, eer_per_system
from src.model import LCNN


def set_all_seeds(seed: int) -> None:
    """Fixe toutes les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_all")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Évalue le modèle sur un DataLoader (dev ou eval).
    Retourne (avg_loss, scores_spoof, labels, systems, utt_ids).
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    scores_list: List[float] = []
    labels_list: List[int] = []
    systems_list: List[str] = []
    utts_list: List[str] = []

    for x, y, systems, utt_ids in loader:
        x = x.to(device)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
        logits = model(x)
        loss = criterion(logits, y_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probabilité de spoof (classe 1)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

        scores_list.extend(probs.cpu().numpy().tolist())
        labels_list.extend(y_tensor.cpu().numpy().tolist())
        systems_list.extend(list(systems))
        utts_list.extend(list(utt_ids))

    avg_loss = total_loss / max(total_count, 1)
    return (
        avg_loss,
        np.array(scores_list, dtype=np.float64),
        np.array(labels_list, dtype=np.int32),
        systems_list,
        utts_list,
    )


def save_scores_csv(
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


def train_single_run(
    condition: str,
    feature_name: str,
    seed: int,
    entries_by_split: Dict[str, List[ProtocolEntry]],
    feature_cache: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    logger: logging.Logger | None = None,
    std_info: Dict | None = None,
    clip: float = 0.0,
) -> Dict:
    """Exécute un run LCNN complet et sauvegarde tous les artefacts requis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)
    device = torch.device("cpu")

    train_ds = WolofAudioDataset(
        entries=entries_by_split["train"],
        feature_cache=feature_cache,
        mean=mean,
        std=std,
        is_train=True,
        seed=seed,
        clip=clip,
    )
    dev_ds = WolofAudioDataset(
        entries=entries_by_split["dev"],
        feature_cache=feature_cache,
        mean=mean,
        std=std,
        is_train=False,
        seed=seed,
        clip=clip,
    )
    eval_ds = WolofAudioDataset(
        entries=entries_by_split["eval"],
        feature_cache=feature_cache,
        mean=mean,
        std=std,
        is_train=False,
        seed=seed,
        clip=clip,
    )

    # Journalisation de la standardisation : max |valeur standardisée| (crops centrés, sans clipping)
    max_abs = {}
    for split_name in ("train", "dev", "eval"):
        probe = WolofAudioDataset(
            entries=entries_by_split[split_name], feature_cache=feature_cache,
            mean=mean, std=std, is_train=False, seed=seed, clip=0.0,
        )
        max_abs[split_name] = round(
            max(float(probe[i][0].abs().max()) for i in range(len(probe))), 3
        )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0
    )
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LCNN(num_classes=2, dropout=0.5).to(device)
    # Poids inverses des effectifs (rapport spoof/bonafide = 3:1 -> poids bonafide=3.0, spoof=1.0)
    class_weights = torch.tensor([3.0, 1.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[Dict] = []
    best_dev_eer = float("inf")
    best_dev_loss = float("inf")
    best_epoch = 1
    best_ckpt_path = out_dir / "best.pt"

    run_start = time.time()
    process = psutil.Process(os.getpid())

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y, _, _ in train_loader:
            x = x.to(device)
            y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        scheduler.step()
        train_loss = train_loss_sum / max(train_count, 1)

        dev_loss, dev_scores, dev_labels, _, _ = evaluate_model(
            model, dev_loader, criterion, device
        )
        dev_eer, dev_thr = compute_eer(dev_scores, dev_labels)
        dev_auc = compute_auc(dev_scores, dev_labels)
        ep_sec = time.time() - ep_t0
        rss_mb = process.memory_info().rss / (1024 * 1024)

        # Sélection du meilleur modèle sur l'EER dev (et dev_loss en cas d'ex-aequo)
        is_better = (dev_eer < best_dev_eer - 1e-6) or (
            abs(dev_eer - best_dev_eer) <= 1e-6 and dev_loss < best_dev_loss
        )
        if is_better:
            best_dev_eer = dev_eer
            best_dev_loss = dev_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "dev_eer": dev_eer,
                    "dev_loss": dev_loss,
                    "dev_threshold": dev_thr,
                    "condition": condition,
                    "feature": feature_name,
                    "seed": seed,
                },
                best_ckpt_path,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "dev_loss": round(dev_loss, 6),
                "dev_eer": round(dev_eer, 4),
                "dev_auc": round(dev_auc, 6),
                "epoch_sec": round(ep_sec, 3),
                "rss_mb": round(rss_mb, 1),
            }
        )

        if logger and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            logger.info(
                f"  [{condition}/{feature_name}/seed{seed}] Ep {epoch:02d}/{epochs:02d} | "
                f"train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | "
                f"dev_EER={dev_eer:.2f}% (best={best_dev_eer:.2f}% @ep{best_epoch}) | "
                f"{ep_sec:.2f}s | RAM={rss_mb:.0f} MB"
            )

    total_train_sec = time.time() - run_start

    # Rechargement du meilleur modèle sélectionné sur dev
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, dev_scores, dev_labels, dev_sys, dev_utts = evaluate_model(
        model, dev_loader, criterion, device
    )
    _, eval_scores, eval_labels, eval_sys, eval_utts = evaluate_model(
        model, eval_loader, criterion, device
    )

    dev_eer, dev_thr = compute_eer(dev_scores, dev_labels)
    dev_auc = compute_auc(dev_scores, dev_labels)

    eval_eer, eval_thr = compute_eer(eval_scores, eval_labels)
    eval_auc = compute_auc(eval_scores, eval_labels)
    eval_sys_eer = eer_per_system(eval_scores, eval_labels, eval_sys)
    eval_boot = bootstrap_ci(eval_scores, eval_labels, n_boot=1000, seed=seed)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_scores_csv(out_dir / "scores_dev.csv", dev_utts, dev_sys, dev_labels, dev_scores)
    save_scores_csv(out_dir / "scores_eval.csv", eval_utts, eval_sys, eval_labels, eval_scores)

    metrics = {
        "condition": condition,
        "feature": feature_name,
        "model": "LCNN",
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
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
        **({} if std_info is None else {
            "std_floor": std_info.get("std_floor"),
            "std_floor_mode": std_info.get("std_floor_mode"),
            "n_bands_floored": std_info.get("n_bands_floored"),
            "min_std": std_info.get("min_std"),
        }),
        "std_clip": clip,
        "max_abs_standardized_train": max_abs["train"],
        "max_abs_standardized_dev": max_abs["dev"],
        "max_abs_standardized_eval": max_abs["eval"],
        "eval_score_std": float(np.std(eval_scores)),
        "eval_n_distinct_scores": int(len(np.unique(np.asarray(eval_scores)))),
        "train_duration_sec": round(total_train_sec, 2),
        "mean_epoch_sec": round(float(np.mean([h["epoch_sec"] for h in history])), 3),
        "peak_rss_mb": round(float(max(h["rss_mb"] for h in history)), 1),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def run_pilot_mode(logger: logging.Logger) -> None:
    """
    TÂCHE 2 — Pilote :
    Lance 1 condition (original) × 1 feature (lfcc) × 1 graine (0) × 3 époques.
    Mesure également la vitesse de calcul/cache pour 10 fichiers CQT afin d'estimer
    avec exactitude le temps de mise en cache + entraînement des 36 runs.
    """
    torch.set_num_threads(4)
    logger.info("=== DÉMARRAGE TÂCHE 2 : PILOTE (original × lfcc × seed 0 × 3 époques) ===")

    t_cache0 = time.time()
    entries_by_split = load_all_splits("original")
    feature_cache, mean, std = extract_and_cache_condition("original", "lfcc", entries_by_split)
    lfcc_cache_sec = time.time() - t_cache0
    logger.info(f"Temps de préparation/cache LFCC (1 012 fichiers) : {lfcc_cache_sec:.2f} s")

    # Mesure sur un échantillon de 20 fichiers pour CQT afin d'estimer le temps du cache CQT
    from src.data import read_audio_sf
    from src.features import extract_feature_from_waveform

    sample_entries = entries_by_split["train"][:20]
    t_cqt0 = time.time()
    for e in sample_entries:
        wav = read_audio_sf(e.wav_path)
        _ = extract_feature_from_waveform(wav, "cqt")
    cqt_20_sec = time.time() - t_cqt0
    est_cqt_per_cond_sec = cqt_20_sec * (1012.0 / 20.0)

    pilot_dir = Path("experiments") / "pilot_task2" / "original" / "lfcc" / "seed0"
    metrics = train_single_run(
        condition="original",
        feature_name="lfcc",
        seed=0,
        entries_by_split=entries_by_split,
        feature_cache=feature_cache,
        mean=mean,
        std=std,
        out_dir=pilot_dir,
        epochs=3,
        batch_size=16,
        logger=logger,
    )

    mean_ep_sec = metrics["mean_epoch_sec"]
    est_single_run_30ep = mean_ep_sec * 30.0
    # Pour mel (128 bandes) et cqt (84 bins), le temps de conv est proportionnel à F
    # F_mfcc=40, F_lfcc=60, F_cqt=84, F_mel=128 -> F_moyen = (40+60+84+128)/4 = 78 = 1.3 * F_lfcc
    est_36_runs_train_sec = 36 * est_single_run_30ep * 1.35
    est_all_caches_sec = 3 * (lfcc_cache_sec * 3 + est_cqt_per_cond_sec)
    est_total_sec = est_36_runs_train_sec + est_all_caches_sec

    summary = {
        "pilot_condition": "original",
        "pilot_feature": "lfcc",
        "pilot_seed": 0,
        "pilot_epochs": 3,
        "mean_epoch_sec": mean_ep_sec,
        "pilot_dev_eer_pct": metrics["dev_eer"],
        "pilot_eval_eer_pct": metrics["eval_eer"],
        "peak_rss_mb": metrics["peak_rss_mb"],
        "lfcc_cache_1012_files_sec": round(lfcc_cache_sec, 2),
        "est_cqt_cache_per_condition_sec": round(est_cqt_per_cond_sec, 1),
        "est_36_runs_training_min": round(est_36_runs_train_sec / 60.0, 1),
        "est_total_with_cache_min": round(est_total_sec / 60.0, 1),
        "est_total_with_cache_hours": round(est_total_sec / 3600.0, 2),
    }

    with open(Path("experiments") / "pilot_task2" / "pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== RÉSUMÉ PILOTE (TÂCHE 2) ===")
    logger.info(f"  - Temps moyen par époque (LFCC) : {mean_ep_sec:.2f} s")
    logger.info(f"  - EER dev après 3 époques       : {metrics['dev_eer']:.2f} %")
    logger.info(f"  - Mémoire RAM pic (RSS)         : {metrics['peak_rss_mb']:.1f} Mo")
    logger.info(
        f"  - Estimation temps total (cache 3 cond × 4 features + 36 runs × 30 époques) : "
        f"~{summary['est_total_with_cache_min']:.1f} minutes (~{summary['est_total_with_cache_hours']:.2f} h)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement LCNN + GMM-LFCC (Wolof Deepfake)")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["original", "matched", "equalized"],
        choices=["original", "matched", "equalized"],
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["mfcc", "lfcc", "mel", "cqt"],
        choices=["mfcc", "lfcc", "mel", "cqt"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--exp-root", type=str, default="experiments")
    parser.add_argument("--pilot", action="store_true", help="Exécute uniquement le pilote TÂCHE 2")
    parser.add_argument("--skip-gmm", action="store_true", help="Ne pas lancer la baseline GMM-LFCC")
    parser.add_argument("--force", action="store_true", help="Recalculer même si metrics.json existe")
    parser.add_argument(
        "--std-floor", choices=["relative", "legacy"], default="relative",
        help="Plancher d'écart-type : relative = max(1e-3*median(sigma), 1e-2) (défaut) ; legacy = 1e-6",
    )
    parser.add_argument(
        "--std-clip", type=float, default=0.0,
        help="Clipper les valeurs standardisées à [-c, c] (0 = désactivé)",
    )
    args = parser.parse_args()

    torch.set_num_threads(4)
    exp_root = Path(args.exp_root)
    logger = setup_logger(exp_root / "train.log")

    if args.pilot:
        run_pilot_mode(logger)
        return

    failed_runs: List[str] = []
    completed_runs: List[str] = []
    results_dir = Path("results")

    import importlib
    make_results_mod = importlib.import_module("05_make_results")

    def _refresh_summary() -> None:
        try:
            rows = make_results_mod.collect_all_metrics(exp_root=exp_root, results_dir=results_dir)
            logger.info(f"  -> results/summary.csv mis à jour ({len(rows)} lignes).")
        except Exception as exc:
            logger.warning(f"  -> Impossible de mettre à jour results/summary.csv : {exc}")

    # Pré-chargement des entrées de protocole par condition
    splits_by_cond: Dict[str, Dict[str, List[ProtocolEntry]]] = {}
    for condition in args.conditions:
        splits_by_cond[condition] = load_all_splits(condition)

    # 1. Baselines GMM-LFCC (une par condition)
    if not args.skip_gmm:
        for condition in args.conditions:
            try:
                logger.info(f"[{condition}/gmm_lfcc] Exécution de la baseline GMM-LFCC...")
                gmm_metrics = run_gmm_baseline(
                    condition=condition, seed=0, exp_root=exp_root, force=args.force
                )
                logger.info(
                    f"[{condition}/gmm_lfcc] Terminé : dev_EER={gmm_metrics['dev_eer']:.2f}%, "
                    f"eval_EER={gmm_metrics['eval_eer']:.2f}%, eval_AUC={gmm_metrics['eval_auc']:.4f}"
                )
                _refresh_summary()
            except Exception as e:
                err_id = f"{condition}/gmm_lfcc/seed0"
                failed_runs.append(err_id)
                logger.error(f"Échec sur {err_id}: {e}\n{traceback.format_exc()}")

    # 2. Boucle principale par GRAINE d'abord, puis condition x feature
    for seed in args.seeds:
        logger.info(f"==================== GRAINE (SEED) : {seed} ====================")
        for condition in args.conditions:
            entries_by_split = splits_by_cond[condition]
            for feature_name in args.features:
                run_id = f"{condition}/{feature_name}/seed{seed}"
                out_dir = exp_root / condition / feature_name / f"seed{seed}"
                metrics_path = out_dir / "metrics.json"

                if metrics_path.exists() and not args.force:
                    logger.info(f"[{run_id}] metrics.json existe déjà -> saut automatique.")
                    completed_runs.append(run_id)
                    _refresh_summary()
                    continue

                try:
                    t_cache = time.time()
                    feature_cache, mean, std = extract_and_cache_condition(
                        condition=condition,
                        feature_name=feature_name,
                        entries_by_split=entries_by_split,
                        std_floor=args.std_floor,
                    )
                    _, std_info = apply_std_floor(
                        np.load(f"experiments/cache/{condition}/{feature_name}/train_stats.npz")["std"],
                        args.std_floor,
                    )
                    cache_dt = time.time() - t_cache
                    if cache_dt > 2.0:
                        logger.info(f"[{condition}/{feature_name}] Cache préparé en {cache_dt:.1f}s")

                    logger.info(f"[{run_id}] Démarrage entraînement LCNN ({args.epochs} époques)...")
                    metrics = train_single_run(
                        condition=condition,
                        feature_name=feature_name,
                        seed=seed,
                        entries_by_split=entries_by_split,
                        feature_cache=feature_cache,
                        mean=mean,
                        std=std,
                        out_dir=out_dir,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        logger=logger,
                        std_info=std_info,
                        clip=args.std_clip,
                    )
                    logger.info(
                        f"[{run_id}] Terminé en {metrics['train_duration_sec']:.1f}s | "
                        f"best_ep={metrics['best_epoch']} | dev_EER={metrics['dev_eer']:.2f}% | "
                        f"eval_EER={metrics['eval_eer']:.2f}% | eval_AUC={metrics['eval_auc']:.4f}"
                    )
                    completed_runs.append(run_id)
                    _refresh_summary()
                except Exception as e:
                    failed_runs.append(run_id)
                    logger.error(f"ÉCHEC du run {run_id} : {e}\n{traceback.format_exc()}")

    logger.info("====================================================")
    logger.info(f"Runs LCNN terminés avec succès : {len(completed_runs)}")
    if failed_runs:
        logger.warning(f"Runs en échec ({len(failed_runs)}) : {failed_runs}")
    else:
        logger.info("Aucun run en échec (0 erreur).")


if __name__ == "__main__":
    main()

