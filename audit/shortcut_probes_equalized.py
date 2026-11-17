"""Etape 1 du plan : sondes de raccourcis sur la condition `equalized` (lecture seule sur data/ et experiments/).
Descripteurs MANUELS (aucune info apprise sur les classes), 3 groupes + tous. Modeles a hyperparametres FIXES
(pas de reglage -> pas de fuite dev/eval). Deux protocoles :
  (a) split officiel : fit sur train, test sur eval (comparable aux detecteurs) ;
  (b) validation croisee par locuteur : 5 plis de 2 locuteurs (data/protocols_new/speaker_folds.json), scores hors-pli poolés.
Controle : etiquettes permutees (AUC attendue ~0,5).
Sorties : audit/probe_features_equalized.csv, audit/shortcut_probes_equalized.csv, audit/shortcut_probes_equalized.json"""
import sys, json, csv, os
import numpy as np
import soundfile as sf
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, '.')
from src.data import load_all_splits

SR = 16000
HOP = 160  # 10 ms
NPERM = 20


def eer(s, y):
    f, t, _ = roc_curve(y, s)
    fn = 1 - t
    i = int(np.argmin(np.abs(f - fn)))
    return float(.5 * (f[i] + fn[i]) * 100)


def feats(path):
    x, sr = sf.read(str(path), dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR
    eps = 1e-10
    o = {}
    # --- G1 : triviaux (duree, niveau, silence, ecretage, DC)
    o['duree_s'] = len(x) / SR
    rms = np.sqrt(np.mean(x ** 2) + eps)
    o['rms_db'] = 20 * np.log10(rms)
    o['peak_db'] = 20 * np.log10(np.abs(x).max() + eps)
    o['crest_db'] = o['peak_db'] - o['rms_db']
    o['clip_frac'] = float((np.abs(x) > 0.99).mean())
    o['dc'] = float(x.mean())
    fr = librosa.feature.rms(y=x, frame_length=400, hop_length=HOP)[0]
    fdb = 20 * np.log10(fr + eps)
    o['silence_frac'] = float((fdb < fdb.max() - 40).mean())
    o['dyn_db'] = float(np.percentile(fdb, 95) - np.percentile(fdb, 10))
    # --- G2 : spectre (LTAS en mel 40 bandes sur frames de parole, centroide, rolloff, platitude, HF)
    S = np.abs(librosa.stft(x, n_fft=512, hop_length=HOP)) ** 2
    speech = fdb[:S.shape[1]] > fdb.max() - 30
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=40, fmax=8000)
    lm = np.log(mel + 1e-10)
    Ls = lm[:, speech[:lm.shape[1]]] if speech.any() else lm
    for i in range(40):
        o[f'ltas_mel{i:02d}'] = float(Ls[i].mean())
    for i in range(40):
        o[f'ltas_std_mel{i:02d}'] = float(Ls[i].std())
    P = S[:, speech[:S.shape[1]]] if speech.any() else S
    fq = librosa.fft_frequencies(sr=SR, n_fft=512)
    ps = P.mean(1)
    o['centroid_hz'] = float((fq * ps).sum() / (ps.sum() + eps))
    c = np.cumsum(ps) / (ps.sum() + eps)
    o['rolloff85_hz'] = float(fq[np.searchsorted(c, .85)])
    o['rolloff95_hz'] = float(fq[np.searchsorted(c, .95)])
    o['flatness'] = float(np.exp(np.mean(np.log(ps + 1e-12))) / (ps.mean() + 1e-12))
    o['hf_4k_db'] = float(10 * np.log10(ps[fq >= 4000].sum() / (ps.sum() + eps) + 1e-12))
    # --- G3 : temporel (flux spectral, spectre de modulation de l'enveloppe)
    Sl = np.log(S + 1e-10)
    flux = np.sqrt((np.diff(Sl, axis=1) ** 2).mean(0))
    o['flux_moy'] = float(flux.mean())
    o['flux_std'] = float(flux.std())
    env = fr - fr.mean()
    M = np.abs(np.fft.rfft(env * np.hanning(len(env)), n=1024)) ** 2
    mf = np.fft.rfftfreq(1024, d=HOP / SR)
    tot = M[(mf >= 0.5) & (mf < 32)].sum() + eps
    for a, b in ((0.5, 2), (2, 4), (4, 8), (8, 16), (16, 32)):
        o[f'mod_{a}_{b}Hz'] = float(M[(mf >= a) & (mf < b)].sum() / tot)
    zc = librosa.feature.zero_crossing_rate(x, frame_length=400, hop_length=HOP)[0]
    o['zcr_moy'] = float(zc.mean())
    o['zcr_std'] = float(zc.std())
    return o


sp = load_all_splits('equalized')
entries = [(s, e) for s in ('train', 'dev', 'eval') for e in sp[s]]
cache = 'audit/probe_features_equalized.csv'
if os.path.exists(cache):
    rows = list(csv.DictReader(open(cache, encoding='utf-8')))
else:
    rows = []
    for k, (s, e) in enumerate(entries):
        r = dict(utt_id=e.utt_id, split=s, spk=e.spk.upper(), system=e.system, y=e.label)
        r.update(feats(e.wav_path))
        rows.append(r)
        if k % 200 == 0:
            print('feat', k, flush=True)
    with open(cache, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
meta = ['utt_id', 'split', 'spk', 'system', 'y']
names = [k for k in rows[0] if k not in meta]
G1 = ['duree_s', 'rms_db', 'peak_db', 'crest_db', 'clip_frac', 'dc', 'silence_frac', 'dyn_db']
G3 = ['flux_moy', 'flux_std', 'mod_0.5_2Hz', 'mod_2_4Hz', 'mod_4_8Hz', 'mod_8_16Hz', 'mod_16_32Hz', 'zcr_moy', 'zcr_std']
G2 = [n for n in names if n not in G1 and n not in G3]
groups = {'G1_triviaux': G1, 'G2_spectre': G2, 'G3_temporel': G3, 'TOUS': names}
X = {n: np.array([float(r[n]) for r in rows]) for n in names}
y = np.array([int(r['y']) for r in rows])
spk = np.array([r['spk'] for r in rows])
sysm = np.array([r['system'] for r in rows])
split = np.array([r['split'] for r in rows])
utt = np.array([r['utt_id'] for r in rows])
folds = json.load(open('data/protocols_new/speaker_folds.json'))['folds']


def models():
    return {'lineaire': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')),
            'arbres': HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.1, random_state=0, class_weight='balanced')}


def summarize(s, yy, sy):
    d = dict(auc=roc_auc_score(yy, s), eer=eer(s, yy))
    for a in ('S01', 'S02', 'S03'):
        m = (sy == a) | (yy == 0)
        d[f'auc_{a}'] = roc_auc_score(yy[m], s[m])
    return d


out, probe_scores = [], {}
rng = np.random.default_rng(12345)
for gname, cols in groups.items():
    M = np.stack([X[c] for c in cols], 1)
    for mname in ('lineaire', 'arbres'):
        # (a) split officiel
        tr, ev = split == 'train', split == 'eval'
        mdl = models()[mname].fit(M[tr], y[tr])
        se = mdl.predict_proba(M[ev])[:, 1]
        d = summarize(se, y[ev], sysm[ev])
        pa = [roc_auc_score(y[ev], models()[mname].fit(M[tr], rng.permutation(y[tr])).predict_proba(M[ev])[:, 1]) for _ in range(NPERM)]
        d['auc_perm_mean'], d['auc_perm_lo'], d['auc_perm_hi'] = float(np.mean(pa)), float(np.percentile(pa, 2.5)), float(np.percentile(pa, 97.5))
        out.append(dict(groupe=gname, modele=mname, protocole='split_officiel', n_features=len(cols), **d))
        probe_scores[(gname, mname)] = dict(zip(utt[ev], se))
        # (b) CV par locuteur
        oof = np.full(len(y), np.nan)
        oofp = np.full((NPERM, len(y)), np.nan)
        fa = []
        for f in folds:
            te = np.isin(spk, [s.upper() for s in f['eval']])
            trn = ~te
            m1 = models()[mname].fit(M[trn], y[trn])
            oof[te] = m1.predict_proba(M[te])[:, 1]
            for q in range(NPERM):
                oofp[q, te] = models()[mname].fit(M[trn], rng.permutation(y[trn])).predict_proba(M[te])[:, 1]
            fa.append(roc_auc_score(y[te], oof[te]))
        d = summarize(oof, y, sysm)
        pa = [roc_auc_score(y, oofp[q]) for q in range(NPERM)]
        d['auc_perm_mean'], d['auc_perm_lo'], d['auc_perm_hi'] = float(np.mean(pa)), float(np.percentile(pa, 2.5)), float(np.percentile(pa, 97.5))
        d['auc_fold_min'], d['auc_fold_max'], d['auc_fold_sd'] = float(min(fa)), float(max(fa)), float(np.std(fa, ddof=1))
        out.append(dict(groupe=gname, modele=mname, protocole='cv_5plis_locuteurs', n_features=len(cols), **d))
        print(gname, mname, 'officiel auc %.3f | cv auc %.3f (plis %.3f-%.3f) | nulle %.3f [%.3f;%.3f]' % (
            out[-2]['auc'], d['auc'], d['auc_fold_min'], d['auc_fold_max'], d['auc_perm_mean'], d['auc_perm_lo'], d['auc_perm_hi']), flush=True)

keys = []
for r in out:
    for k in r:
        if k not in keys:
            keys.append(k)
with open('audit/shortcut_probes_equalized.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for r in out:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

# --- comparaison au detecteur sur l'eval officielle (ensemble des 3 graines = moyenne des scores)
det = {}
for ft in ('mfcc', 'lfcc', 'mel', 'cqt'):
    ss = []
    for s in (0, 1, 2):
        rr = list(csv.DictReader(open(f'experiments/equalized/{ft}/seed{s}/scores_eval.csv', encoding='utf-8')))
        ss.append({r['utt_id']: float(r['score']) for r in rr})
    det[ft] = {u: float(np.mean([d[u] for d in ss])) for u in ss[0]}
ev_u = [u for u in utt[split == 'eval']]
yev = y[split == 'eval']
best = max((v for v in out if v['protocole'] == 'split_officiel'), key=lambda v: v['auc'])
bp = probe_scores[(best['groupe'], best['modele'])]
sb = np.array([bp[u] for u in ev_u])
i0, i1 = np.where(yev == 0)[0], np.where(yev == 1)[0]
cmp_ = dict(meilleure_sonde=dict(groupe=best['groupe'], modele=best['modele'], auc=best['auc'], eer=best['eer']), detecteurs={})
for ft, d in det.items():
    sd = np.array([d[u] for u in ev_u])
    dE, dA = [], []
    r2 = np.random.default_rng(12345)
    for _ in range(1000):
        idx = np.concatenate([r2.choice(i0, len(i0)), r2.choice(i1, len(i1))])
        dE.append(eer(sd[idx], yev[idx]) - eer(sb[idx], yev[idx]))
        dA.append(roc_auc_score(yev[idx], sd[idx]) - roc_auc_score(yev[idx], sb[idx]))
    cmp_['detecteurs'][ft] = dict(auc=roc_auc_score(yev, sd), eer=eer(sd, yev), dEER_vs_sonde=eer(sd, yev) - eer(sb, yev),
                                  dEER_ci95=[float(np.percentile(dE, 2.5)), float(np.percentile(dE, 97.5))],
                                  dAUC_vs_sonde=roc_auc_score(yev, sd) - roc_auc_score(yev, sb),
                                  dAUC_ci95=[float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))])
json.dump(dict(sondes=out, comparaison=cmp_), open('audit/shortcut_probes_equalized.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(cmp_, indent=1, ensure_ascii=False))
