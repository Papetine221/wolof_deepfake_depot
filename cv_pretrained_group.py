"""[VARIANTE GROUPES] Retire de l'entrainement (et de la CV interne) tout clip dont la voix (groupe de personnes ECAPA, seuil 0.5) est presente
dans le test. Variantes : orig, grp. Sorties : experiments_pretrained/equalized/cv_speakers_grp/. Ancien docstring :
Controle de fuite S02 : la validation croisee par locuteur est refaite sous 4 variantes.
S02 (XTTS, voix d'un AUTRE locuteur du meme split d'origine) a pour reference le clip spkXX_ref, aussi utilise pour les S01 de ce locuteur.
En CV, la reference d'un S02 de test peut donc etre un locuteur d'ENTRAINEMENT (timbre clone deja vu), et reciproquement un S02
d'entrainement peut cloner un locuteur de TEST.
Variantes :
  orig       : protocole de cv_pretrained.py (reference)
  testS02out : S02 retire du TEST ; entrainement inchange
  clean      : S02 retire du TEST ; S02 d'entrainement dont la reference est un locuteur de TEST retire de l'entrainement
  noS02      : S02 retire de l'entrainement ET du test (S01 + S03 + bona fide seulement)
Le meme filtrage est applique dans la CV interne de selection (sous-plis) et a la sonde de raccourcis. Sorties :
experiments_pretrained/equalized/cv_speakers_s02/{results.json, oof_<modele>_<variante>.csv}"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

CS = (1e-2, 1e-1, 1.0)
NB, RNG0 = 1000, 12345
ROOT = Path('experiments_pretrained/equalized')
OUT = ROOT / 'cv_speakers_grp'
OUT.mkdir(exist_ok=True)
folds = json.load(open('data/protocols_new/speaker_folds.json'))['folds']
man = json.load(open('data/manifest_stm.json'))['records']
REF = {r['utt_id']: r['ref_s02_spk'] for r in man}
VARIANTS = ('orig', 'grp')
import sys; sys.path.insert(0, 'audit'); import leak_groups as LG
LAB = LG.labels(0.5)
GRP = {}


def eer_thr(s, y):
    f, t, th = roc_curve(y, s)
    fn = 1 - t
    i = int(np.argmin(np.abs(f - fn)))
    return float(.5 * (f[i] + fn[i]) * 100), float(th[i])


def eer(s, y):
    return eer_thr(s, y)[0]


def fit_score(Xtr, ytr, Xs, C):
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    lr = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit((Xtr - mu) / sd, ytr)
    return [lr.decision_function((X - mu) / sd) for X in Xs]


def filt(trm, tem, variant, spk, sysm, refspk):
    """Retourne (masque entrainement, masque test) apres filtrage de la variante."""
    tspk = set(spk[tem])
    trm, tem = trm.copy(), tem.copy()
    if variant in ('testS02out', 'clean', 'noS02'):
        tem &= sysm != 'S02'
    if variant == 'clean':
        trm &= ~((sysm == 'S02') & np.isin(refspk, list(tspk)))
    if variant == 'noS02':
        trm &= sysm != 'S02'
    if variant == 'grp':
        tg = set().union(*[GRP[u] for u in UTT[tem]]) if tem.any() else set()
        trm &= np.array([not (GRP[u] & tg) for u in UTT])
    return trm, tem


def inner_cfg(X, y, spk, sysm, refspk, trspk, layer, pool, C, variant):
    sc = np.full(len(y), np.nan)
    for k in range(4):
        va = np.isin(spk, trspk[2 * k:2 * k + 2])
        tr, va = filt(np.isin(spk, trspk) & ~va, va, variant, spk, sysm, refspk)
        sc[va] = fit_score(X[tr], y[tr], [X[va]], C)[0]
    m = ~np.isnan(sc)
    return layer, pool, C, eer(sc[m], y[m]), roc_auc_score(y[m], sc[m])


def features(d, layer, pool):
    m = np.load(d / 'mean.npy', mmap_mode='r')[:, layer]
    if pool == 'mean':
        return np.asarray(m, np.float64)
    return np.concatenate([m, np.load(d / 'std.npy', mmap_mode='r')[:, layer]], 1).astype(np.float64)


res = {}
import builtins
sh_rows = list(csv.DictReader(open('audit/probe_features_equalized.csv', encoding='utf-8')))
names = [k for k in sh_rows[0] if k not in ('utt_id', 'split', 'spk', 'system', 'y')]
SX = {r['utt_id']: [float(r[n]) for n in names] for r in sh_rows}

for key in ('wavlm', 'wav2vec2'):
    d = ROOT / key
    idx = list(csv.DictReader(open(d / 'index.csv', encoding='utf-8')))
    y = np.array([int(r['y']) for r in idx])
    spk = np.array([r['spk'] for r in idx])
    sysm = np.array([r['system'] for r in idx])
    utt = np.array([r['utt_id'] for r in idx])
    UTT = utt
    GRP = {u: LG.clip_groups(u, LAB) for u in utt}
    refspk = np.array([REF.get(u.replace('S02_', '', 1), '') if s == 'S02' else '' for u, s in zip(utt, sysm)])
    Xs = np.array([SX[u] for u in utt])
    FX = {(L, p): features(d, L, p) for L in range(13) for p in ('mean', 'mean+std')}
    for variant in VARIANTS:
        t0 = time.time()
        oof = np.full(len(y), np.nan)
        osh = np.full(len(y), np.nan)
        rows = []
        for f in folds:
            te0 = np.isin(spk, [s.upper() for s in f['eval']])
            trspk = sorted(s.upper() for s in f['train'])
            trm, te = filt(~te0, te0, variant, spk, sysm, refspk)
            jobs = [(L, p, C) for L in range(13) for p in ('mean', 'mean+std') for C in CS]
            outs = Parallel(n_jobs=6)(delayed(inner_cfg)(FX[(L, p)], y, spk, sysm, refspk, trspk, L, p, C, variant) for L, p, C in jobs)
            L, p, C, ieer, _ = min(outs, key=lambda o: (o[3], -o[4]))
            oof[te] = fit_score(FX[(L, p)][trm], y[trm], [FX[(L, p)][te]], C)[0]
            osh[te] = fit_score(Xs[trm], y[trm], [Xs[te]], 1.0)[0]
            rows.append(dict(fold=f['fold'], layer=L, pool=p, C=C, inner_eer=ieer, n_train=int(trm.sum()), n_test=int(te.sum()),
                             test_eer=eer(oof[te], y[te]), test_auc=float(roc_auc_score(y[te], oof[te])), shortcut_eer=eer(osh[te], y[te])))
        ev = ~np.isnan(oof)
        r = dict(per_fold=rows, mean_fold_eer=float(np.mean([x['test_eer'] for x in rows])), sd_fold_eer=float(np.std([x['test_eer'] for x in rows], ddof=1)),
                 mean_fold_auc=float(np.mean([x['test_auc'] for x in rows])), pooled_eer=eer(oof[ev], y[ev]),
                 shortcut_mean_fold_eer=float(np.mean([x['shortcut_eer'] for x in rows])), shortcut_pooled_eer=eer(osh[ev], y[ev]),
                 per_attack_pooled_eer={a: eer(oof[ev & ((sysm == a) | (y == 0))], y[ev & ((sysm == a) | (y == 0))]) for a in ('S01', 'S02', 'S03') if (ev & (sysm == a)).any()},
                 n_test_files=int(ev.sum()))
        # S02 selon que la reference est cote entrainement (variante orig uniquement : S02 present en test)
        if variant == 'orig':
            tr_ref = np.zeros(len(y), bool)
            for f in folds:
                te0 = np.isin(spk, [s.upper() for s in f['eval']])
                tr_ref |= te0 & (sysm == 'S02') & ~np.isin(refspk, [s.upper() for s in f['eval']])
            r['n_S02_test_ref_in_train'] = int(tr_ref.sum())
            r['n_S02_test_total'] = int((sysm == 'S02').sum())
        res[f'{key}/{variant}'] = r
        with open(OUT / f'oof_{key}_{variant}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['utt_id', 'spk', 'system', 'label', 'score'])
            for u, s_, a, b, c in zip(utt[ev], spk[ev], sysm[ev], y[ev], oof[ev]):
                w.writerow([u, s_, a, 'spoof' if b else 'bonafide', repr(float(c))])
        print('%-9s %-10s EER par pli %s -> moyenne %.2f ± %.2f | poolé %.2f | sonde raccourcis %.2f | par attaque %s | n_test %d | %.0fs' % (
            key, variant, [round(x['test_eer'], 1) for x in rows], r['mean_fold_eer'], r['sd_fold_eer'], r['pooled_eer'], r['shortcut_mean_fold_eer'],
            {a: round(v, 1) for a, v in r['per_attack_pooled_eer'].items()}, r['n_test_files'], time.time() - t0), flush=True)
        json.dump(res, open(OUT / 'results.json', 'w'), indent=1)
