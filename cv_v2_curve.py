"""Courbe d'apprentissage : sonde WavLM (couche 4, mean+std, C=0.1 fixes) ; on sous-echantillonne les PERSONNES d'entrainement (25/50/75/100 %), 8 tirages, test = pli complet inchange."""
import csv, json, sys
from pathlib import Path
import numpy as np
from cv_v2_probe import eer, fit_score
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'wavlm'; LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 4; CC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
d = Path('experiments_pretrained/v2/' + MODEL)
idx = list(csv.DictReader(open(d/'index.csv', encoding='utf-8')))
y = np.array([int(r['y']) for r in idx]); fold = np.array([int(r['fold']) for r in idx])
pers = np.array([r['person'] for r in idx]); sysm = np.array([r['system'] for r in idx])
m = np.load(d/'mean.npy'); s = np.load(d/'std.npy')
X = np.concatenate([m[:, LAYER], s[:, LAYER]], 1).astype(np.float64)
res = {}
for frac in (0.25, 0.5, 0.75, 1.0):
    per_fold = []
    for f in range(5):
        te = fold == f
        trp = sorted(set(pers[fold != f])); reps = 1 if frac == 1.0 else 8
        e = []
        for r in range(reps):
            rng = np.random.default_rng(r)
            keep = set(rng.choice(trp, max(2, round(frac*len(trp))), replace=False))
            tr = np.isin(pers, list(keep)) & (fold != f)
            e.append(eer(fit_score(X[tr], y[tr], X[te], CC), y[te]))
        per_fold.append(float(np.mean(e)))
    res[frac] = dict(n_pers=round(frac*len(trp)), mean=float(np.mean(per_fold)), sd=float(np.std(per_fold, ddof=1)), per_fold=[round(v, 2) for v in per_fold])
    print(frac, res[frac], flush=True)
json.dump({str(k): v for k, v in res.items()}, open('experiments_pretrained/v2/cv/' + ('curve.json' if MODEL == 'wavlm' else 'curve_%s.json' % MODEL), 'w'), indent=1)
