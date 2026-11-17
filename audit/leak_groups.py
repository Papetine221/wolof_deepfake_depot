"""Utilitaire : groupes de personnes (embeddings ECAPA) -> ensemble de groupes par clip (bona fide / S01 / S02 / S03)."""
import csv, json, re
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

OUT = 'audit/spk_embed'
_ent = list(csv.DictReader(open(f'{OUT}/entries.csv', encoding='utf-8')))
_E = np.load(f'{OUT}/emb_mean.npy')
_Z = linkage(squareform(np.clip(1 - _E @ _E.T, 0, None), checks=False), 'average')
_ix = {(e['file'], e['spk']): i for i, e in enumerate(_ent)}
_rec = {r['utt_id']: r for r in json.load(open('data/manifest_stm.json'))['records']}


def labels(thr):
    return fcluster(_Z, 1 - thr, 'distance')


def clip_groups(utt, lab):
    """Groupes de personnes d'un clip : sa propre entree + la/les voix de reference clonees (S01 : ref du locuteur ; S02 : ref d'un autre)."""
    base = re.sub(r'^S0\d_', '', utt)
    r = _rec[base]
    keys = [r]
    if utt.startswith('S01_') and r.get('ref_s01') in _rec:
        keys.append(_rec[r['ref_s01']])
    if utt.startswith('S02_') and r.get('ref_s02') in _rec:
        keys.append(_rec[r['ref_s02']])
    out = set()
    for k in keys:
        i = _ix.get((k['file_id'], k['locuteur'].replace('locuteur_', 'spk')))
        if i is not None:
            out.add(int(lab[i]))
    return out
