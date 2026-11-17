"""Figures du memoire (base v2) -> memoire_latex/figures/*.pdf (+ png de controle dans audit/figures_apercu/). Chiffres lus dans les fichiers de resultats."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

OUT = Path('memoire_latex/figures'); OUT.mkdir(exist_ok=True)
PREV = Path('audit/figures_apercu'); PREV.mkdir(exist_ok=True)
CV = Path('experiments_pretrained/v2/cv')
R = json.load(open(CV / 'results.json'))
for _k in ('xlsr', 'mms'):
    R.update(json.load(open(CV / f'results_{_k}.json')))
L = json.load(open('experiments_pretrained/v2/lcnn/cv_summary_plis.json'))
CUR = json.load(open(CV / 'curve_xlsr.json'))
COL = {'raccourcis': '#8a8f98', 'LCNN': '#d9822b', 'wav2vec 2.0': '#2a9d8f', 'WavLM': '#1f4e9c', 'MMS': '#8e4ca8', 'XLS-R': '#c2385f'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f'{name}.pdf'); fig.savefig(PREV / f'{name}.png', dpi=130); plt.close(fig)

def labels(ax, bars, vals, fmt='%.1f', err=None):
    err = err if err is not None else [0] * len(vals)
    for b, v, e in zip(bars, vals, err):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.5, fmt % v, ha='center', va='bottom', fontsize=8.5)

def foldeer(path):
    d = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        d.setdefault(int(r['fold']), ([], []))[0].append(int(r['label'] == 'spoof')); d[int(r['fold'])][1].append(float(r['score']))
    e = []
    for f in sorted(d):
        fp, tp, _ = roc_curve(np.array(d[f][0]), np.array(d[f][1])); i = int(np.argmin(np.abs(fp - (1 - tp)))); e.append(50 * (fp[i] + 1 - tp[i]))
    return dict(mean_fold_eer=float(np.mean(e)), sd_fold_eer=float(np.std(e, ddof=1)))
def sm(k, fam):
    if k.startswith('manuel'):
        proto = k.split('|')[1]; return foldeer(CV / ('oof_manuel_plis_tous.csv' if proto == 'plis' else f'oof_manuel_logo_{fam}.csv'))
    return R[k]['summary'][fam]
lcnn = [x['test_eer'] for x in L.values()]
# 1) EER en plis
names = ['raccourcis', 'LCNN', 'wav2vec 2.0', 'WavLM', 'MMS', 'XLS-R']
m = [sm('manuel|plis', 'tous')['mean_fold_eer'], np.mean(lcnn), sm('wav2vec2|plis', 'tous')['mean_fold_eer'], sm('wavlm|plis', 'tous')['mean_fold_eer'], sm('mms|plis', 'tous')['mean_fold_eer'], sm('xlsr|plis', 'tous')['mean_fold_eer']]
s = [sm('manuel|plis', 'tous')['sd_fold_eer'], np.std(lcnn, ddof=1), sm('wav2vec2|plis', 'tous')['sd_fold_eer'], sm('wavlm|plis', 'tous')['sd_fold_eer'], sm('mms|plis', 'tous')['sd_fold_eer'], sm('xlsr|plis', 'tous')['sd_fold_eer']]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
b = ax.bar(['Sonde de\nraccourcis', 'LCNN-Mel', 'wav2vec 2.0\nbase gelé', 'WavLM\nbase gelé', 'MMS 300M\ngelé', 'XLS-R 300M\ngelé'], m, yerr=s, capsize=4, color=[COL[n] for n in names], width=0.64)
labels(ax, b, m, '%.2f', s)
ax.set_ylabel('EER moyen par pli (%)'); ax.set_ylim(0, 26)
save(fig, 'fig_eer_plis')
# 2) generateur laisse de cote
fams = [('SpeechT5', 'SpeechT5 (S03)'), ('XTTS', 'XTTS (S01+S02)'), ('kNNVC', 'kNN-VC (S05)')]
fig, ax = plt.subplots(figsize=(7.2, 3.6)); w = 0.16
for j, (k, n) in enumerate((('manuel', 'raccourcis'), ('wav2vec2', 'wav2vec 2.0'), ('wavlm', 'WavLM'), ('mms', 'MMS'), ('xlsr', 'XLS-R'))):
    v = [sm(f'{k}|logo', f)['mean_fold_eer'] for f, _ in fams]; e = [sm(f'{k}|logo', f)['sd_fold_eer'] for f, _ in fams]
    bb = ax.bar(np.arange(3) + (j - 2) * w, v, w, yerr=e, capsize=3, color=COL[n], label={'raccourcis': 'Sonde de raccourcis', 'wav2vec 2.0': 'wav2vec 2.0 base', 'WavLM': 'WavLM base', 'MMS': 'MMS 300M', 'XLS-R': 'XLS-R 300M'}[n])
    labels(ax, bb, v, '%.1f', e)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in fams]); ax.set_ylabel('EER moyen par pli (%)'); ax.set_xlabel('Famille de faux absente de l\'entraînement')
ax.set_ylim(0, 56); ax.legend(frameon=False, loc='upper left', fontsize=8.5, ncol=2)
save(fig, 'fig_eer_logo')
# 3) courbe d'apprentissage
_pf = __import__('collections').Counter(v['fold'] for v in json.load(open('data_v2/manifest_v2.json', encoding='utf-8'))['persons'].values())
_np = sum(_pf.values())
# nombre moyen de personnes d'entrainement sur les plis (n_pers ne donne que le dernier pli)
xs = [float(np.mean([max(2, round(float(k) * (_np - _pf[f]))) for f in range(5)])) for k in ('0.25', '0.5', '0.75', '1.0')]; ym = [CUR[k]['mean'] for k in ('0.25', '0.5', '0.75', '1.0')]; ys = [CUR[k]['sd'] for k in ('0.25', '0.5', '0.75', '1.0')]
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.errorbar(xs, ym, yerr=ys, marker='o', color=COL['XLS-R'], capsize=4, lw=2)
for x, y_ in zip(xs, ym): ax.annotate('%.2f' % y_, (x, y_), textcoords='offset points', xytext=(6, 7), fontsize=8.5)
ax.set_xlabel('Personnes d\'entraînement'); ax.set_ylabel('EER moyen par pli (%)'); ax.set_xlim(8, 66); ax.set_ylim(0, 4.5)
save(fig, 'fig_courbe_apprentissage')
# 4) par systeme
sy = ['S01', 'S02', 'S03', 'S05']
def lcnn_s05():
    e = []
    for f in range(5):
        rr = [r for r in csv.DictReader(open(f'experiments_pretrained/v2/lcnn/plis/tous/fold{f}/scores_eval.csv', encoding='utf-8')) if r['system'] in ('-', 'S05')]
        s = np.array([float(r['score']) for r in rr]); y_ = np.array([int(r['system'] == 'S05') for r in rr])
        fp, tp, _ = roc_curve(y_, s); i = int(np.argmin(np.abs(fp - (1 - tp)))); e.append(50 * (fp[i] + 1 - tp[i]))
    return float(np.mean(e))
rows = {'LCNN': [np.mean([x['eer_per_system'][t] for x in L.values()]) for t in sy[:3]] + [lcnn_s05()],
        'wav2vec 2.0': [sm('wav2vec2|plis', 'tous')['mean_fold_eer_' + t] for t in sy], 'WavLM': [sm('wavlm|plis', 'tous')['mean_fold_eer_' + t] for t in sy],
        'MMS': [sm('mms|plis', 'tous')['mean_fold_eer_' + t] for t in sy], 'XLS-R': [sm('xlsr|plis', 'tous')['mean_fold_eer_' + t] for t in sy]}
fig, ax = plt.subplots(figsize=(7.4, 3.6)); w = 0.16
for j, n in enumerate(rows):
    v = rows[n]; bb = ax.bar(np.arange(4) + (j - 2) * w, np.nan_to_num(v), w, color=COL[n], label=n)
    labels(ax, [b_ for b_, x in zip(bb, v) if not np.isnan(x)], [x for x in v if not np.isnan(x)])
ax.set_xticks(range(4)); ax.set_xticklabels(['S01\nXTTS, même voix', 'S02\nXTTS, autre voix', 'S03\nSpeechT5', 'S05\nkNN-VC'], fontsize=9)
ax.set_ylabel('EER moyen par pli (%)'); ax.set_ylim(0, 14.5); ax.legend(frameon=False, loc='upper center', ncol=5, fontsize=8)
save(fig, 'fig_eer_systeme')
# 5) ROC WavLM, scores hors pli agreges
def roc(path):
    y, s = [], []
    for r in csv.DictReader(open(path, encoding='utf-8')):
        y.append(int(r['label'] == 'spoof')); s.append(float(r['score']))
    return roc_curve(np.array(y), np.array(s))
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for tag, nm, c in (('plis_tous', 'Plis, tous systèmes vus', '#1f4e9c'), ('logo_SpeechT5', 'SpeechT5 inconnu', '#2a9d8f'), ('logo_XTTS', 'XTTS inconnu', '#d9822b'), ('logo_kNNVC', 'kNN-VC inconnu', '#b5333f')):
    f, t, _ = roc(CV / f'oof_wavlm_{tag}.csv'); ax.plot(f * 100, t * 100, color=c, lw=2, label=nm)
ax.plot([0, 100], [100, 0], color='#888', lw=0.8, ls=':'); ax.set_xlabel('Faux rejets (authentiques jugés faux), %'); ax.set_ylabel('Détection des faux, %')
ax.set_xlim(0, 60); ax.set_ylim(40, 100); ax.legend(frameon=False, loc='lower right', fontsize=8.5)
save(fig, 'fig_roc_wavlm')
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for tag, nm, c in (('plis_tous', 'Plis, tous systèmes vus', '#1f4e9c'), ('logo_SpeechT5', 'SpeechT5 inconnu', '#2a9d8f'), ('logo_XTTS', 'XTTS inconnu', '#d9822b'), ('logo_kNNVC', 'kNN-VC inconnu', '#b5333f')):
    f, t, _ = roc(CV / f'oof_xlsr_{tag}.csv'); ax.plot(f * 100, t * 100, color=c, lw=2, label=nm)
ax.plot([0, 100], [100, 0], color='#888', lw=0.8, ls=':'); ax.set_xlabel('Faux rejets (authentiques jugés faux), %'); ax.set_ylabel('Détection des faux, %')
ax.set_xlim(0, 60); ax.set_ylim(40, 100); ax.legend(frameon=False, loc='lower right', fontsize=8.5)
save(fig, 'fig_roc_xlsr')
print('ok', sorted(p.name for p in OUT.glob('*.pdf')))
