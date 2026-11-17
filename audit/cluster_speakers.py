"""Regroupement des (fichier, locuteur local) de speech_dataset en 'personnes' par similarite cosinus des embeddings ECAPA.
Calibrage du seuil :
 - NEGATIFS certains : deux locuteurs d'un MEME fichier = deux personnes differentes (hors enregistrement a plusieurs micros identiques).
 - POSITIFS approches : moitie A vs moitie B des segments d'une meme entree (meme personne, meme fichier => borne haute optimiste).
Le seuil est choisi pour que le taux de fusion erronee (negatifs au-dessus du seuil) soit <= 1 % ; liaison moyenne (average linkage).
Sorties : audit/spk_embed/clusters.csv, cluster_report.json"""
import csv, json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

OUT = 'audit/spk_embed'
ent = list(csv.DictReader(open(f'{OUT}/entries.csv', encoding='utf-8')))
E = np.load(f'{OUT}/emb_mean.npy'); A = np.load(f'{OUT}/emb_halfA.npy'); B = np.load(f'{OUT}/emb_halfB.npy')
n = len(ent)
file = np.array([e['file'] for e in ent]); sec = np.array([float(e['seconds']) for e in ent]); nseg = np.array([int(e['n_seg']) for e in ent])
gender = np.array([e['gender'] for e in ent])
S = E @ E.T
iu = np.triu_indices(n, 1)
same_file = (file[:, None] == file[None, :])[iu]
neg = S[iu][same_file]
pos = np.array([float(a @ b) for a, b in zip(A, B) if not np.isnan(a).any()])
print('entrees', n, 'negatifs (meme fichier)', len(neg), 'positifs (moities)', len(pos))
print('sim negatifs : moy %.3f, p99 %.3f, max %.3f' % (neg.mean(), np.percentile(neg, 99), neg.max()))
print('sim positifs : moy %.3f, p5 %.3f, min %.3f' % (pos.mean(), np.percentile(pos, 5), pos.min()))
thr_neg99 = float(np.percentile(neg, 99))
rep = dict(n_entries=n, n_files=len(set(file)), neg_p99=thr_neg99, neg_max=float(neg.max()), pos_mean=float(pos.mean()), pos_p5=float(np.percentile(pos, 5)), thresholds={})
D = squareform(np.clip(1 - S, 0, None), checks=False)
Z = linkage(D, 'average')
for thr in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
    lab = fcluster(Z, 1 - thr, 'distance')
    # un cluster contenant 2 entrees d'un meme fichier = fusion certainement erronee
    bad = 0
    for c in set(lab):
        f = file[lab == c]
        bad += len(f) - len(set(f))
    sizes = np.bincount(lab)[1:]
    big = sizes >= 2
    rep['thresholds'][str(thr)] = dict(n_clusters=int(len(set(lab))), n_multi=int(big.sum()), max_size=int(sizes.max()), entries_in_multi=int(sizes[big].sum()),
                                       impossible_merges=int(bad))
    print('seuil cos %.2f : %d personnes (%d regroupent >=2 entrees, max %d, entrees fusionnees %d) | fusions impossibles (2 locuteurs du meme fichier) : %d' % (
        thr, len(set(lab)), big.sum(), sizes.max(), sizes[big].sum(), bad))
# seuil retenu : le plus bas seuil sans fusion impossible et >= p99 des negatifs
cand = [float(t) for t in rep['thresholds'] if rep['thresholds'][t]['impossible_merges'] == 0 and float(t) >= thr_neg99 - 1e-9]
thr = min(cand) if cand else 0.8
lab = fcluster(Z, 1 - thr, 'distance')
rep['chosen_threshold'] = thr; rep['n_persons'] = int(len(set(lab)))
print('SEUIL RETENU', thr, '-> personnes', rep['n_persons'])
with open(f'{OUT}/clusters.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['dir', 'file', 'spk', 'gender', 'seconds', 'n_seg', 'person'])
    for e, l in zip(ent, lab):
        w.writerow([e['dir'], e['file'], e['spk'], e['gender'], e['seconds'], e['n_seg'], f'P{l:04d}'])
# hauteur de 'ambiguite' : paires inter-fichiers entre p99(neg) et seuil
json.dump(rep, open(f'{OUT}/cluster_report.json', 'w'), indent=1)
