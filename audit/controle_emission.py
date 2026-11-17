"""Controle « emission vue / non vue » (critique 2 de audit/revue_jury.md).
Les plis separent les personnes, pas les emissions : un authentique de test peut venir d'une emission (show_id)
ou d'un fichier (file_id) dont d'autres segments sont a l'entrainement. On compare, sans reentrainement, a partir
des scores hors pli du protocole a 5 plis :
  - le FRR (authentiques rejetes) au seuil fixe (seuil d'EER des 4 autres plis, comme audit/metrics_extra.py) ;
  - l'AUC contre les faux du meme pli ;
pour les authentiques de test dont l'emission (ou le fichier) est vue a l'entrainement, et pour les autres.
Usage : python audit/controle_emission.py   Sortie : audit/controle_emission.json"""
import csv, json, re
from pathlib import Path
import numpy as np
from scipy.stats import beta, fisher_exact
from sklearn.metrics import roc_auc_score, roc_curve


# memes conventions que audit/metrics_extra.py (recopiees pour ne pas executer ce script a l'import)
def key_of(u, sy):
    base = re.sub(r'^S0\d_', '', u.split('|')[0])
    return base, ('bonafide' if sy in ('-', 'bonafide') else sy)


def eer_thr(s, y):
    f, t, th = roc_curve(y, s)
    i = int(np.argmin(np.abs(f - (1 - t))))
    return float(th[i])

CV = Path('experiments_pretrained/v2/cv')
MODS = ['manuel', 'wav2vec2', 'wavlm', 'mms', 'xlsr']

man = json.load(open('data_v2/manifest_v2.json', encoding='utf-8'))
rec = {r['utt_id']: r for r in man['records']}


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(100 * float(lo), 1), round(100 * float(hi), 1)]


def seen_sets(level):
    """pour chaque pli de test f : ensemble des show_id / file_id presents chez les authentiques des 4 autres plis."""
    return {f: {r[level] for r in man['records'] if r['fold'] != f} for f in range(5)}


out = {}
for level in ('show_id', 'file_id'):
    seen = seen_sets(level)
    out[level] = {}
    for m in MODS:
        rows = list(csv.DictReader(open(CV / f'oof_{m}_plis_tous.csv', encoding='utf-8')))
        s = np.array([float(r['score']) for r in rows]); y = np.array([int(r['label'] == 'spoof') for r in rows])
        fold = np.array([int(r['fold']) for r in rows])
        base = [key_of(r['utt_id'], r['system'])[0] for r in rows]
        vu = np.array([y[i] == 0 and rec[base[i]][level] in seen[fold[i]] for i in range(len(rows))])
        nv = (y == 0) & ~vu
        rej = np.zeros(len(rows), bool); aucs = {'vu': [], 'non_vu': []}
        for f in range(5):
            thr = eer_thr(s[fold != f], y[fold != f]); te = fold == f
            rej[te] = s[te] > thr  # spoof positif : un authentique est rejete si son score depasse le seuil
            for g, msk in (('vu', vu), ('non_vu', nv)):
                b = te & msk; sp = te & (y == 1)
                if b.sum() >= 3:
                    aucs[g].append(roc_auc_score(np.r_[np.zeros(b.sum()), np.ones(sp.sum())], np.r_[s[b], s[sp]]))
        d = {}
        for g, msk in (('vu', vu), ('non_vu', nv)):
            k, n = int((rej & msk).sum()), int(msk.sum())
            d[g] = dict(n=n, rejetes=k, FRR=round(100 * k / max(n, 1), 1), IC95=cp(k, n), AUC_moy_plis=round(float(np.mean(aucs[g])), 4) if aucs[g] else None)
        tab = [[d['vu']['rejetes'], d['vu']['n'] - d['vu']['rejetes']], [d['non_vu']['rejetes'], d['non_vu']['n'] - d['non_vu']['rejetes']]]
        d['p_fisher'] = round(float(fisher_exact(tab)[1]), 3)
        out[level][m] = d
        print(f"{level:7s} {m:8s} | vu : {d['vu']['rejetes']:3d}/{d['vu']['n']:3d} = {d['vu']['FRR']:5.1f} % {d['vu']['IC95']} AUC {d['vu']['AUC_moy_plis']}"
              f" | non vu : {d['non_vu']['rejetes']:3d}/{d['non_vu']['n']:3d} = {d['non_vu']['FRR']:5.1f} % {d['non_vu']['IC95']} AUC {d['non_vu']['AUC_moy_plis']} | Fisher p = {d['p_fisher']}", flush=True)
json.dump(out, open('audit/controle_emission.json', 'w'), indent=1)
