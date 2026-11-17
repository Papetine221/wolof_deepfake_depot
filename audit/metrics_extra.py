"""Metriques complementaires (sans reentrainement) a partir des scores hors pli deja enregistres.
 1) Intervalles de confiance a 95 % par bootstrap SUR LES PERSONNES (reechantillonnage des personnes, avec remise, a l'interieur de chaque pli ;
    1000 tirages, graine fixe) pour l'EER moyen par pli, l'AUC moyenne par pli et des DIFFERENCES entre modeles (memes tirages : apparie).
 2) Metriques a SEUIL FIXE : seuil = seuil d'EER calcule sur les scores hors pli des AUTRES plis (aucune donnee du pli de test), puis FAR, FRR,
    precision, rappel, F1 (classe spoof positive) et exactitude equilibree sur le pli de test.
 3) Generateur laisse de cote : meme seuil, calcule sur les vrais et les systemes VUS des autres plis ; on mesure le FAR sur la famille INCONNUE
    et on le compare au FAR sur les systemes vus du meme pli.
Entrees : experiments_pretrained/v2/cv/oof_*.csv, oof_full_*_logo_*.csv, experiments_pretrained/v2/lcnn/plis/tous/fold*/scores_eval.csv
Sortie : audit/metrics_extra.json, audit/metrics_extra.out"""
import csv, json, re, sys
XL = 'xl' in sys.argv[1:]
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve

CV = Path('experiments_pretrained/v2/cv')
MOD = ['manuel', 'lcnn', 'wav2vec2', 'wavlm'] + (['xlsr', 'mms'] if XL else [])
B = 1000
rng = np.random.default_rng(12345)
FAMS = {'SpeechT5': ('S03',), 'XTTS': ('S01', 'S02'), 'kNNVC': ('S05',)}


def key_of(u, sy):
    base = re.sub(r'^S0\d_', '', u.split('|')[0])
    return base, ('bonafide' if sy in ('-', 'bonafide') else sy)


def person_of(base):
    return base.split('_')[0].upper()


def eer_auc(s, y):
    f, t, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs(f - (1 - t))))
    return 50 * (f[i] + 1 - t[i]), float(np.trapezoid(t, f))


def eer_thr(s, y):
    f, t, th = roc_curve(y, s)
    i = int(np.argmin(np.abs(f - (1 - t))))
    return float(th[i])


def fixed(s, y, thr):
    """spoof positif ; decision spoof si score > seuil."""
    pred = s > thr
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    far = fn / max(fn + tp, 1)      # spoof accepte comme authentique
    frr = fp / max(fp + tn, 1)      # authentique rejete
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(FAR=100 * far, FRR=100 * frr, precision=100 * prec, rappel=100 * rec, F1=100 * f1, exactitude_equilibree=100 * (1 - (far + frr) / 2))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load_scores():
    D = {}
    for m in [x for x in MOD if x != 'lcnn']:
        for r in csv.DictReader(open(CV / f'oof_{m}_plis_tous.csv', encoding='utf-8')):
            k = key_of(r['utt_id'], r['system']); d = D.setdefault(k, dict(fold=int(r['fold']), y=int(r['label'] == 'spoof'), sys=k[1]))
            d[m] = float(r['score'])
    for f in range(5):
        for r in csv.DictReader(open(f'experiments_pretrained/v2/lcnn/plis/tous/fold{f}/scores_eval.csv', encoding='utf-8')):
            k = key_of(r['utt_id'], r['system']); D[k]['lcnn'] = float(r['score'])
    keys = sorted(k for k, d in D.items() if all(m in d for m in MOD))
    assert len(keys) == len(D), (len(keys), len(D))
    return keys, D


def boot_indices(fold, pers):
    """pour chaque pli : liste des tableaux d'indices par personne."""
    out = {}
    for f in range(5):
        m = np.where(fold == f)[0]
        pp = {}
        for i in m:
            pp.setdefault(pers[i], []).append(i)
        out[f] = [np.array(v) for v in pp.values()]
    return out


def bootstrap_draw(groups):
    n = len(groups)
    sel = rng.integers(0, n, n)
    return np.concatenate([groups[j] for j in sel])


lines = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)


# =========================== plis ===========================
keys, D = load_scores()
y = np.array([D[k]['y'] for k in keys]); fold = np.array([D[k]['fold'] for k in keys]); sysm = np.array([D[k]['sys'] for k in keys])
pers = np.array([person_of(k[0]) for k in keys]); S = {m: np.array([D[k][m] for k in keys]) for m in MOD}
thr = {m: {f: eer_thr(S[m][fold != f], y[fold != f]) for f in range(5)} for m in MOD}
groups = boot_indices(fold, pers)
P('PLIS : %d lignes, %d personnes, plis %s' % (len(y), len(set(pers)), {f: len(set(pers[fold == f])) for f in range(5)}))


def stats_plis(idx_by_fold):
    r = {m: dict(eer=[], auc=[], fixed=[]) for m in MOD}
    for f in range(5):
        i = idx_by_fold[f]
        for m in MOD:
            e, a = eer_auc(S[m][i], y[i]); r[m]['eer'].append(e); r[m]['auc'].append(a)
            r[m]['fixed'].append(fixed(S[m][i], y[i], thr[m][f]))
    out = {}
    for m in MOD:
        out[m] = dict(eer=float(np.mean(r[m]['eer'])), auc=float(np.mean(r[m]['auc'])))
        for k in r[m]['fixed'][0]:
            out[m][k] = float(np.mean([x[k] for x in r[m]['fixed']]))
    return out


full = stats_plis({f: np.where(fold == f)[0] for f in range(5)})
reps = []
for b in range(B):
    reps.append(stats_plis({f: bootstrap_draw(groups[f]) for f in range(5)}))
RES = {'plis': {}}
P('\n== 1) Plis de personnes : EER, AUC (IC 95 % bootstrap sur les personnes) ==')
for m in MOD:
    e = [r[m]['eer'] for r in reps]; a = [r[m]['auc'] for r in reps]
    RES['plis'][m] = dict(eer=full[m]['eer'], eer_ci=ci(e), auc=full[m]['auc'], auc_ci=ci(a))
    P('%-9s EER %.2f [%.2f ; %.2f]   AUC %.4f [%.4f ; %.4f]' % (m, full[m]['eer'], *ci(e), full[m]['auc'], *ci(a)))
P('\n== 1b) Differences d\'EER (memes tirages) : IC 95 %, et part des tirages ou la difference est de signe oppose ==')
RES['plis_diff'] = {}
for a_, b_ in (('wavlm', 'lcnn'), ('wavlm', 'wav2vec2'), ('wav2vec2', 'lcnn'), ('lcnn', 'manuel'), ('wavlm', 'manuel')) + ((('xlsr', 'wavlm'), ('mms', 'wavlm'), ('xlsr', 'mms'), ('xlsr', 'lcnn'), ('mms', 'lcnn'), ('xlsr', 'wav2vec2')) if XL else ()):
    dv = np.array([r[a_]['eer'] - r[b_]['eer'] for r in reps]); d0 = full[a_]['eer'] - full[b_]['eer']
    opp = float(np.mean(dv >= 0) if d0 < 0 else np.mean(dv <= 0))
    RES['plis_diff'][f'{a_}-{b_}'] = dict(diff=d0, ci=ci(dv), p_oppose=opp)
    P('%-9s - %-9s : %+.2f [%+.2f ; %+.2f]   (tirages de signe oppose : %.3f)' % (a_, b_, d0, *ci(dv), opp))
P('\n== 2) Seuil fixe (seuil d\'EER des autres plis ; moyenne par pli ; IC 95 %) ==')
RES['plis_fixed'] = {}
for m in MOD:
    row = {}
    for k in ('FAR', 'FRR', 'precision', 'rappel', 'F1', 'exactitude_equilibree'):
        v = [r[m][k] for r in reps]; row[k] = dict(val=full[m][k], ci=ci(v))
    RES['plis_fixed'][m] = row
    P('%-9s ' % m + ' | '.join('%s %.1f [%.1f ; %.1f]' % (k, row[k]['val'], *row[k]['ci']) for k in row))
# FAR par systeme au seuil fixe (valeur ponctuelle)
P('\n   FAR par systeme au seuil fixe (moyenne par pli, sans IC) :')
RES['plis_far_systeme'] = {}
for m in MOD:
    d = {}
    for sy in ('S01', 'S02', 'S03', 'S05'):
        d[sy] = float(np.mean([100 * np.mean(S[m][(fold == f) & (sysm == sy)] <= thr[m][f]) for f in range(5)]))
    RES['plis_far_systeme'][m] = d
    P('  %-9s %s' % (m, {k: round(v, 1) for k, v in d.items()}))

# =========================== generateur laisse de cote ===========================
P('\n== 3) Generateur laisse de cote : seuil fixe calcule sur vrais + systemes VUS des autres plis ==')
RES['logo'] = {}
for m in ('manuel', 'wav2vec2', 'wavlm') + (('xlsr', 'mms') if XL else ()):
    for fam, drop in FAMS.items():
        rows = list(csv.DictReader(open(CV / f'oof_full_{m}_logo_{fam}.csv', encoding='utf-8')))
        ks = [key_of(r['utt_id'], r['system']) for r in rows]
        yy = np.array([int(r['label'] == 'spoof') for r in rows]); ff = np.array([int(r['fold']) for r in rows]); sc = np.array([float(r['score']) for r in rows])
        seen = np.array([int(r['seen']) for r in rows]).astype(bool); sy = np.array([r['system'] for r in rows])
        pp = np.array([person_of(k[0]) for k in ks])
        held = (sy == 'bonafide') | np.isin(sy, drop)           # test : vrais + famille inconnue
        seen_rows = (sy == 'bonafide') | seen                    # reference : vrais + systemes vus
        th = {f: eer_thr(sc[(ff != f) & seen_rows], yy[(ff != f) & seen_rows]) for f in range(5)}
        gr = boot_indices(ff, pp)

        def stat(idx):
            e_, far_u, frr_, far_s = [], [], [], []
            for f in range(5):
                i = idx[f]; ih = i[held[i]]; isn = i[seen_rows[i]]
                e_.append(eer_auc(sc[ih], yy[ih])[0])
                fx = fixed(sc[ih], yy[ih], th[f]); far_u.append(fx['FAR']); frr_.append(fx['FRR'])
                far_s.append(fixed(sc[isn], yy[isn], th[f])['FAR'])
            return np.mean(e_), np.mean(far_u), np.mean(frr_), np.mean(far_s)
        base = stat({f: np.where(ff == f)[0] for f in range(5)})
        bs = np.array([stat({f: bootstrap_draw(gr[f]) for f in range(5)}) for _ in range(B)])
        RES['logo'][f'{m}|{fam}'] = dict(eer=base[0], eer_ci=ci(bs[:, 0]), far_inconnu=base[1], far_inconnu_ci=ci(bs[:, 1]), frr=base[2], frr_ci=ci(bs[:, 2]),
                                          far_vus=base[3], far_vus_ci=ci(bs[:, 3]))
        P('%-9s %-9s EER %.1f [%.1f ; %.1f] | FAR famille inconnue %.1f [%.1f ; %.1f] | FAR systemes vus %.1f [%.1f ; %.1f] | FRR %.1f [%.1f ; %.1f]' % (
            m, fam, base[0], *ci(bs[:, 0]), base[1], *ci(bs[:, 1]), base[3], *ci(bs[:, 3]), base[2], *ci(bs[:, 2])))
json.dump(RES, open('audit/metrics_extra%s.json' % ('_xl' if XL else ''), 'w'), indent=1)
open('audit/metrics_extra%s.out' % ('_xl' if XL else ''), 'w', encoding='utf-8').write('\n'.join(lines))
