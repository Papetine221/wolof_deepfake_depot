"""Test Oolel (S04) avec la sonde XLS-R 300M, configuration fixe de la courbe d'apprentissage (couche 7, mean+std, C=1).
Meme protocole que audit/test_s04_oolel.py (WavLM) : les 8 clips deja egalises (data_v2/pilot_equalized), chaque clip note par un
modele entraine SANS le pli de sa personne ; seuil = seuil EER des scores hors-pli de la base principale.
Lancer depuis la racine : python audit/test_s04_oolel_xlsr.py"""
import glob, os, csv, re, json, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from scipy.stats import beta
from transformers import AutoModel
from sklearn.metrics import roc_curve, roc_auc_score
from cv_v2_probe import fit_score, eer
from src.data import read_audio_sf, crop_and_peak_normalize
LAYER, CC = 7, 1.0
man = {r['utt_id']: r for r in json.load(open('data_v2/manifest_v2.json', encoding='utf-8'))['records']}
clips = []
for p in sorted(glob.glob('data_v2/pilot_equalized/spoof/S04_oolel_wolof/*.wav')):
    u = re.sub(r'^S04_', '', os.path.basename(p)[:-4])
    clips.append((u, p, man[u]['fold']))
print('clips S04 egalises', len(clips), [(u, f) for u, _, f in clips])
model = AutoModel.from_pretrained('pretrained_models/facebook__wav2vec2-xls-r-300m').eval()
def feat(path):
    a = crop_and_peak_normalize(read_audio_sf(path), mode='center').astype(np.float64)
    a = (a - a.mean()) / np.sqrt(a.var() + 1e-7)  # comme extract_pretrained_v2.py (do_norm)
    with torch.no_grad():
        h = model(torch.from_numpy(a[None].astype(np.float32)), output_hidden_states=True).hidden_states[LAYER][0]
    return np.concatenate([h.mean(0).numpy(), h.std(0).numpy()]).astype(np.float64)
d = 'experiments_pretrained/v2/xlsr'
idx = list(csv.DictReader(open(d + '/index.csv', encoding='utf-8')))
y = np.array([int(r['y']) for r in idx]); fold = np.array([int(r['fold']) for r in idx])
X = np.concatenate([np.load(d + '/mean.npy')[:, LAYER], np.load(d + '/std.npy')[:, LAYER]], 1).astype(np.float64)
# controle : la fonction feat reproduit bien les caracteristiques enregistrees
r0 = idx[0]; f0 = feat(r0['path']); print('controle extraction, ecart max', float(np.abs(f0 - X[0]).max()))
oof = np.zeros(len(y))
for f in range(5):
    oof[fold == f] = fit_score(X[fold != f], y[fold != f], X[fold == f], CC)
fp, tp, th = roc_curve(y, oof); i = int(np.argmin(np.abs(fp - (1 - tp)))); thr = th[i]
print('seuil EER hors-pli %.3f (EER %.2f %%)' % (thr, eer(oof, y)))
sc = np.array([float(fit_score(X[fold != f], y[fold != f], feat(p)[None], CC)[0]) for u, p, f in clips])
bsc = oof[y == 0]
k, n = int((sc > thr).sum()), len(sc)
lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
print('scores S04', np.round(sc, 2).tolist())
print('detectes comme faux au seuil EER : %d / %d ; au seuil 0 : %d / %d' % (k, n, (sc > 0).sum(), n))
print('Clopper-Pearson 95 %% : %.1f a %.1f %%' % (100 * lo, 100 * hi))
print('AUC S04 vs vrais (scores hors-pli) : %.3f' % roc_auc_score(np.r_[np.zeros(len(bsc)), np.ones(n)], np.r_[bsc, sc]))
print('EER S04 vs vrais : %.2f %%' % eer(np.r_[bsc, sc], np.r_[np.zeros(len(bsc)), np.ones(n)]))
