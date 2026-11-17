"""Scores COMPLETS du protocole 'generateur laisse de cote' : on refit, pour chaque (famille, pli), le modele avec la configuration deja choisie
(experiments_pretrained/v2/cv/results.json, aucune nouvelle selection) et l'on enregistre les scores de TOUTES les lignes du pli de test
(vrais, systemes vus ET famille laissee de cote). Sert au calcul d'un seuil sur les systemes vus (audit/metrics_extra.py).
Sortie : experiments_pretrained/v2/cv/oof_full_<modele>_logo_<famille>.csv (colonnes utt_id, system, fold, label, score, seen)
Controle : l'EER de la famille laissee de cote doit retrouver celui de results.json."""
import csv, json, sys
from pathlib import Path
import numpy as np
from cv_v2_probe import FAM, eer, fit_score

CV = Path('experiments_pretrained/v2/cv')
KEYS = sys.argv[1:] or ['wavlm', 'wav2vec2']
R = {}
for k_ in KEYS:
    R.update(json.load(open(CV / ('results.json' if k_ in ('wavlm', 'wav2vec2') else 'results_%s.json' % k_))))
for key in KEYS:
    d = Path('experiments_pretrained/v2/' + key)
    idx = list(csv.DictReader(open(d / 'index.csv', encoding='utf-8')))
    y = np.array([int(r['y']) for r in idx]); fold = np.array([int(r['fold']) for r in idx])
    sysm = np.array([r['system'] for r in idx]); utt = np.array([r['utt_id'] for r in idx])
    mean = np.load(d / 'mean.npy'); sd = np.load(d / 'std.npy')
    NL = mean.shape[1]
    FX = {(L, 'mean'): mean[:, L].astype(np.float64) for L in range(NL)}
    FX.update({(L, 'mean+std'): np.concatenate([mean[:, L], sd[:, L]], 1).astype(np.float64) for L in range(NL)})
    cfg = {(r['family'], r['fold']): r['cfg'] for r in R[key + '|logo']['rows']}
    chk = {(r['family'], r['fold']): r['eer'] for r in R[key + '|logo']['rows']}
    for fam, drop in FAM.items():
        out = []
        for f in range(5):
            L, p, C = cfg[(fam, f)]
            tr = (fold != f) & ~np.isin(sysm, list(drop)); te = fold == f
            s = fit_score(FX[(L, p)][tr], y[tr], FX[(L, p)][te], C)
            held = (sysm[te] == 'bonafide') | np.isin(sysm[te], list(drop))
            e = eer(s[held], y[te][held])
            assert abs(e - chk[(fam, f)]) < 1e-6, (key, fam, f, e, chk[(fam, f)])
            for u, sy, yy, sc in zip(utt[te], sysm[te], y[te], s):
                out.append((u, sy, f, 'spoof' if yy else 'bonafide', repr(float(sc)), int(sy not in drop)))
        with open(CV / f'oof_full_{key}_logo_{fam}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh); w.writerow(['utt_id', 'system', 'fold', 'label', 'score', 'seen']); w.writerows(out)
        print(key, fam, 'ok, EER famille laissee de cote identique a results.json', flush=True)

if sys.argv[1:]:
    sys.exit(0)
# sonde de raccourcis (caracteristiques manuelles, configuration fixe C=1)
rows0 = list(csv.DictReader(open('audit/v2_probe_features_eq.csv', encoding='utf-8')))
names = [k for k in rows0[0] if k not in ('sy', 'utt', 'person', 'fold', 'y')]
y = np.array([int(r['y']) for r in rows0]); fold = np.array([int(r['fold']) for r in rows0]); sysm = np.array([r['sy'] for r in rows0])
S = np.array([[float(r[n]) for n in names] for r in rows0]); utt = np.array([r['utt'] + '|' + r['sy'] for r in rows0])
for fam, drop in FAM.items():
    out = []
    for f in range(5):
        tr = (fold != f) & ~np.isin(sysm, list(drop)); te = fold == f
        s = fit_score(S[tr], y[tr], S[te], 1.0)
        for u, sy, yy, sc in zip(utt[te], sysm[te], y[te], s):
            out.append((u, sy, f, 'spoof' if yy else 'bonafide', repr(float(sc)), int(sy not in drop)))
    with open(CV / f'oof_full_manuel_logo_{fam}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['utt_id', 'system', 'fold', 'label', 'score', 'seen']); w.writerows(out)
    print('manuel', fam, 'ok', flush=True)
