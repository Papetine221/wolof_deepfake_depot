"""Sondes de raccourcis sur la base v2 (memes caracteristiques que shortcut_probes_equalized.py), validation par PLI de personnes (5 plis, groupes ECAPA).
Usage : python audit/v2_probes.py <racine> (ex. data_v2/eq) ; les faux sont lus sous <racine>/spoof (egalise) ou data_v2/gen (brut, --raw)."""
import sys, glob, json, os, re, csv
import numpy as np, soundfile as sf, librosa
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
src = open('audit/shortcut_probes_equalized.py', encoding='utf-8').read().split('\n')
exec('\n'.join(src[18:next(i for i, l in enumerate(src) if l.startswith('sp = load_all_splits'))]))  # SR, HOP, eer, feats
tag = sys.argv[1]
if tag == 'eq':
    pats = [('data_v2/eq/bonafide/*.wav', 'bonafide')] + [(f'data_v2/eq/spoof/{s}*/*.wav', s[:3]) for s in ('S01', 'S02', 'S03', 'S05')]
else:
    pats = [('data_v2/real/bonafide/*.wav', 'bonafide')] + [(f'data_v2/gen/{s}*/*.wav', s[:3]) for s in ('S01', 'S02', 'S03', 'S05')]
man = {r['utt_id']: r for r in json.load(open('data_v2/manifest_v2.json', encoding='utf-8'))['records']}
files = [(p, sy) for pat, sy in pats for p in glob.glob(pat)]
cache = f'audit/v2_probe_features_{tag}.csv'
if os.path.exists(cache):
    rows = list(csv.DictReader(open(cache, encoding='utf-8')))
    for r in rows:
        for k in r:
            if k not in ('sy', 'utt', 'person', 'fold'):
                r[k] = float(r[k])
else:
    def f1(p, sy):
        u = re.sub(r'^S0\d_', '', os.path.basename(p)[:-4])
        d = feats(p); d.update(sy=sy, utt=u, person=man[u]['person'], fold=man[u]['fold'], y=int(sy != 'bonafide')); return d
    rows = Parallel(n_jobs=4)(delayed(f1)(p, sy) for p, sy in files)
    with open(cache, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
names = [k for k in rows[0] if k not in ('sy', 'utt', 'person', 'fold', 'y')]
G1 = ['duree_s', 'rms_db', 'peak_db', 'crest_db', 'clip_frac', 'dc', 'silence_frac', 'dyn_db']
G3 = ['flux_moy', 'flux_std', 'mod_0.5_2Hz', 'mod_2_4Hz', 'mod_4_8Hz', 'mod_8_16Hz', 'mod_16_32Hz', 'zcr_moy', 'zcr_std']
G2 = [n for n in names if n not in G1 and n not in G3]
y = np.array([int(r['y']) for r in rows]); fold = np.array([int(r['fold']) for r in rows]); sysm = np.array([r['sy'] for r in rows])
print(tag, 'n', len(y), {s: int((sysm == s).sum()) for s in ('bonafide', 'S01', 'S02', 'S03', 'S05')}, flush=True)
out = {}
for gn, g in (('G1 triviaux', G1), ('G2 spectre', G2), ('G3 temporel', G3), ('TOUS', names)):
    X = np.array([[float(r[n]) for n in g] for r in rows])
    for mn, mk in (('lineaire', lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced'))),
                   ('arbres', lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=0, class_weight='balanced'))):
        sc = np.zeros(len(y))
        for f in range(5):
            te = fold == f
            sc[te] = mk().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        per = {s: round(roc_auc_score(y[(sysm == s) | (sysm == 'bonafide')], sc[(sysm == s) | (sysm == 'bonafide')]), 2) for s in ('S01', 'S02', 'S03', 'S05')}
        out[f'{gn}|{mn}'] = dict(auc_all=round(roc_auc_score(y, sc), 3), per_system=per)
        print('%-12s %-8s AUC (hors pli) tous faux %.2f | par systeme %s' % (gn, mn, roc_auc_score(y, sc), per), flush=True)
json.dump(out, open(f'audit/v2_probes_{tag}.json', 'w'), indent=1)
