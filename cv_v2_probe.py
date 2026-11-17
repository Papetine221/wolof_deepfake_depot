"""Sonde lineaire gelee (WavLM / wav2vec2) sur la base v2, deux protocoles, selection (couche, pooling, C) par CV INTERNE sur les plis d'entrainement :
  A) 'plis'  : 5 plis de personnes, tous les systemes en entrainement ;
  B) 'logo'  : un generateur (famille) laisse de cote : XTTS (S01+S02), SpeechT5 (S03), kNN-VC (S05) ; test = personnes du pli + famille laissee de cote.
Reference : sonde de raccourcis (caracteristiques manuelles, audit/v2_probe_features_eq.csv) avec les MEMES protocoles.
Usage : python cv_v2_probe.py [wavlm] [wav2vec2] [manuel]
Sorties : experiments_pretrained/v2/cv/{results.json, oof_<modele>_<protocole>_<famille>.csv}"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

NL = 13
CS = (1e-2, 1e-1, 1.0)
FAM = {'XTTS': ('S01', 'S02'), 'SpeechT5': ('S03',), 'kNNVC': ('S05',)}
OUT = Path('experiments_pretrained/v2/cv')
OUT.mkdir(parents=True, exist_ok=True)


def eer(s, y):
    f, t, _ = roc_curve(y, s)
    fn = 1 - t
    i = int(np.argmin(np.abs(f - fn)))
    return float(.5 * (f[i] + fn[i]) * 100)


def fit_score(Xtr, ytr, Xte, C):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight='balanced')).fit(Xtr, ytr)
    return m.decision_function(Xte)


def inner(FX, y, fold, sysm, trfolds, layer, pool, C, drop_sys):
    """CV interne : chaque pli d'entrainement tour a tour comme validation (memes systemes exclus de l'entrainement)."""
    sc = np.full(len(y), np.nan)
    drop = list(drop_sys)
    for vf in trfolds:
        tr = np.isin(fold, [f for f in trfolds if f != vf]) & ~np.isin(sysm, drop)
        va = (fold == vf)
        sc[va] = fit_score(FX[(layer, pool)][tr], y[tr], FX[(layer, pool)][va], C)
    m = ~np.isnan(sc)
    return layer, pool, C, eer(sc[m], y[m]), roc_auc_score(y[m], sc[m])


def run(FX, y, fold, sysm, protocol, shortcut=None):
    """Retourne (scores hors-pli par famille, lignes par (pli, famille))."""
    fams = [(None, ())] if protocol == 'plis' else list(FAM.items())
    oof = {}
    rows = []
    for fam, drop in fams:
        drop = list(drop)
        sc = np.full(len(y), np.nan)
        for f in range(5):
            trf = [g for g in range(5) if g != f]
            tr = np.isin(fold, trf) & ~np.isin(sysm, drop)
            te = (fold == f) & ((sysm == 'bonafide') | np.isin(sysm, drop)) if drop else (fold == f)
            if shortcut is not None:
                s = fit_score(shortcut[tr], y[tr], shortcut[te], 1.0)
                cfg = ('manuel', '-', 1.0)
            else:
                jobs = [(L, p, C) for L in range(NL) for p in ('mean', 'mean+std') for C in CS]
                outs = Parallel(n_jobs=6)(delayed(inner)(FX, y, fold, sysm, trf, L, p, C, drop) for L, p, C in jobs)
                L, p, C, _, _ = min(outs, key=lambda o: (o[3], -o[4]))
                cfg = (L, p, C)
                s = fit_score(FX[(L, p)][tr], y[tr], FX[(L, p)][te], C)
            sc[te] = s
            r = dict(fold=f, family=fam or 'tous', cfg=cfg, n_train=int(tr.sum()), n_test=int(te.sum()), eer=eer(s, y[te]), auc=float(roc_auc_score(y[te], s)))
            for sy in ('S01', 'S02', 'S03', 'S05'):
                m2 = te & ((sysm == sy) | (sysm == 'bonafide'))
                if (sysm[te] == sy).any():
                    r['eer_' + sy] = eer(sc[m2], y[m2])
            rows.append(r)
        oof[fam or 'tous'] = sc
    return oof, rows


def summarize(rows):
    out = {}
    for fam in sorted({r['family'] for r in rows}):
        rr = [r for r in rows if r['family'] == fam]
        d = dict(mean_fold_eer=float(np.mean([r['eer'] for r in rr])), sd_fold_eer=float(np.std([r['eer'] for r in rr], ddof=1)),
                 mean_fold_auc=float(np.mean([r['auc'] for r in rr])), eer_per_fold=[round(r['eer'], 1) for r in rr])
        for sy in ('S01', 'S02', 'S03', 'S05'):
            v = [r['eer_' + sy] for r in rr if r.get('eer_' + sy) is not None]
            if v:
                d['mean_fold_eer_' + sy] = float(np.mean(v))
        out[fam] = d
    return out


if __name__ == '__main__':
    models = sys.argv[1:] or ['wavlm', 'wav2vec2', 'manuel']
    res = {}
    for key in models:
        S = None
        FX = None
        if key == 'manuel':
            rows0 = list(csv.DictReader(open('audit/v2_probe_features_eq.csv', encoding='utf-8')))
            names = [k for k in rows0[0] if k not in ('sy', 'utt', 'person', 'fold', 'y')]
            y = np.array([int(r['y']) for r in rows0])
            fold = np.array([int(r['fold']) for r in rows0])
            sysm = np.array([r['sy'] for r in rows0])
            S = np.array([[float(r[n]) for n in names] for r in rows0])
            utt = np.array([r['utt'] + '|' + r['sy'] for r in rows0])
        else:
            d = Path('experiments_pretrained/v2/' + key)
            idx = list(csv.DictReader(open(d / 'index.csv', encoding='utf-8')))
            y = np.array([int(r['y']) for r in idx])
            fold = np.array([int(r['fold']) for r in idx])
            sysm = np.array([r['system'] for r in idx])
            utt = np.array([r['utt_id'] for r in idx])
            mean = np.load(d / 'mean.npy')
            sd = np.load(d / 'std.npy')
            NL = mean.shape[1]
            FX = {(L, 'mean'): mean[:, L].astype(np.float64) for L in range(NL)}
            FX.update({(L, 'mean+std'): np.concatenate([mean[:, L], sd[:, L]], 1).astype(np.float64) for L in range(NL)})
        for proto in ('plis', 'logo'):
            t0 = time.time()
            oof, rows = run(FX, y, fold, sysm, proto, shortcut=S)
            sm = summarize(rows)
            res[key + '|' + proto] = dict(summary=sm, rows=rows)
            for fam, v in sm.items():
                print('%-9s %-5s %-9s EER moyen par pli %.2f +- %.2f | AUC %.4f | par pli %s | par systeme %s | %.0fs' % (
                    key, proto, fam, v['mean_fold_eer'], v['sd_fold_eer'], v['mean_fold_auc'], v['eer_per_fold'],
                    {k[-3:]: round(x, 1) for k, x in v.items() if k.startswith('mean_fold_eer_')}, time.time() - t0), flush=True)
            for fam, sc in oof.items():
                with open(OUT / ('oof_%s_%s_%s.csv' % (key, proto, fam)), 'w', newline='', encoding='utf-8') as fh:
                    w = csv.writer(fh)
                    w.writerow(['utt_id', 'system', 'fold', 'label', 'score'])
                    for u, s_, f_, y_, c in zip(utt, sysm, fold, y, sc):
                        if not np.isnan(c):
                            w.writerow([u, s_, f_, 'spoof' if y_ else 'bonafide', repr(float(c))])
        json.dump(res, open(OUT / ('results.json' if set(models) <= {'wavlm', 'wav2vec2', 'manuel'} else 'results_' + '_'.join(models) + '.json'), 'w'), indent=1, default=str)
