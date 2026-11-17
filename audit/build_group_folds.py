"""(1) Fuite dans le split actuel a 10 'locuteurs' ; (2) construction de plis par GROUPES de personnes (embeddings ECAPA) pour speech_dataset.
Sorties : protocols_groups/persons_t050.csv (entree -> personne), protocols_groups/folds_t050.json (5 plis + split train/dev/test),
audit/leak_current_split.json. N'ecrit rien dans data/, experiments/, results/."""
import csv, json, glob, os, collections, xml.etree.ElementTree as ET
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

OUT = 'audit/spk_embed'
ent = list(csv.DictReader(open(f'{OUT}/entries.csv', encoding='utf-8')))
E = np.load(f'{OUT}/emb_mean.npy'); S = E @ E.T
Z = linkage(squareform(np.clip(1 - S, 0, None), checks=False), 'average')
file = np.array([e['file'] for e in ent]); spk = np.array([e['spk'] for e in ent])
# duree totale de parole mono-locuteur par entree (tous tours)
tot = collections.defaultdict(float)
for trs in glob.glob('speech_dataset/*/*.trs'):
    f = os.path.basename(trs)[:-4]
    for t in ET.parse(trs).getroot().iter('Turn'):
        s = (t.get('speaker') or '').split()
        if len(s) == 1:
            tot[(f, s[0])] += float(t.get('endTime')) - float(t.get('startTime'))
sec = np.array([tot[(f, s)] for f, s in zip(file, spk)])
gender = np.array([e['gender'] for e in ent])

# ---- (1) fuite du split actuel
m = json.load(open('data/manifest_stm.json')); split_of = m['split_of']
cur = collections.defaultdict(set)
for r in m['records']:
    cur[r['spk_id']].add((r['file_id'], r['locuteur'].replace('locuteur_', 'spk')))
ix = {(a, b): i for i, (a, b) in enumerate(zip(file, spk))}
leak = {}
for thr in (0.5, 0.6, 0.65, 0.7, 0.8):
    lab = fcluster(Z, 1 - thr, 'distance')
    byc = collections.defaultdict(lambda: collections.defaultdict(set))
    for s, ks in cur.items():
        for k in ks:
            if k in ix:
                byc[lab[ix[k]]][split_of[s]].add(s)
    leaked = {sp: sorted({s for c, d in byc.items() if sp in d and len(d) > 1 for s in d[sp]}) for sp in ('train', 'dev', 'eval')}
    nc = sum(1 for d in byc.values() if len(d) > 1)
    leak[str(thr)] = dict(clusters_crossing_splits=nc, speakers_with_cross_split_voice=leaked)
    print('seuil', thr, 'groupes a cheval sur plusieurs splits :', nc, '| locuteurs concernes', leaked)
json.dump(leak, open('audit/leak_current_split.json', 'w'), indent=1)

# ---- (2) plis par groupes (seuil 0.60 = sur-fusion volontaire)
THR = 0.50
lab = fcluster(Z, 1 - THR, 'distance')
with open('protocols_groups/persons_t050.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['dir', 'file', 'spk', 'gender', 'seconds_total', 'person'])
    for e, s_, l in zip(ent, sec, lab):
        w.writerow([e['dir'], e['file'], e['spk'], e['gender'], round(float(s_), 1), f'P{l:04d}'])
groups = collections.defaultdict(list)
for i, l in enumerate(lab):
    groups[int(l)].append(i)
G = [dict(id=int(g), n=len(ix_), sec=float(sec[ix_].sum()), fem=float(sec[ix_][gender[ix_] == 'female'].sum())) for g, ix_ in groups.items()]
G.sort(key=lambda x: -x['sec'])
# greedy : chaque groupe va dans le pli le plus en retard (duree), les groupes avec femmes sont repartis en priorite
K = 5; fold = {k: dict(groups=[], sec=0.0, fem=0.0, n=0) for k in range(K)}
for g in sorted(G, key=lambda x: (-(x['fem'] > 0), -x['sec'])):
    k = min(fold, key=lambda k: (fold[k]['fem'] if g['fem'] > 0 else fold[k]['sec'], fold[k]['sec']))
    fold[k]['groups'].append(g['id']); fold[k]['sec'] += g['sec']; fold[k]['fem'] += g['fem']; fold[k]['n'] += g['n']
for k, f in fold.items():
    print('pli', k, 'groupes', len(f['groups']), 'entrees', f['n'], 'heures %.1f' % (f['sec'] / 3600), 'dont femmes %.1f h' % (f['fem'] / 3600))
# split fixe train/dev/test = plis (0,1,2)=train, 3=dev, 4=test
res = dict(threshold=THR, n_persons=len(G), folds={str(k): dict(person_groups=[f'P{g:04d}' for g in f['groups']], entries=f['n'], hours=f['sec'] / 3600, female_hours=f['fem'] / 3600)
                                                  for k, f in fold.items()},
           fixed_split=dict(train=[0, 1, 2], dev=[3], test=[4]))
json.dump(res, open('protocols_groups/folds_t050.json', 'w'), indent=1)
# verification : aucun groupe n'est dans 2 plis ; liens residuels entre plis a un seuil plus strict (0.5) et plus lache (0.8)
pf = {}
for k, f in fold.items():
    for g in f['groups']:
        for i in groups[g]:
            pf[i] = k
for t2 in (0.45, 0.4, 0.35):
    l2 = fcluster(Z, 1 - t2, 'distance')
    cross = sum(1 for c in set(l2) if len({pf[i] for i in np.where(l2 == c)[0]}) > 1)
    print('controle : groupes (seuil %.2f) a cheval sur plusieurs plis : %d / %d' % (t2, cross, len(set(l2))))
