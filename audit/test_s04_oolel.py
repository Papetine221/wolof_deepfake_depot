"""Test du modele (sonde WavLM, couche 4 mean+std, C=0.1) sur S04 Oolel, generateur JAMAIS vu. Seuls les 8 clips valides de l'essai pilote existent.
Chaque clip est note par un modele entraine SANS le pli de sa personne ; seuil = seuil EER des scores hors-pli (vrais vs faux S01/S02/S03/S05)."""
import glob, os, csv, re, json, random, zlib
import numpy as np, soundfile as sf, torch
from scipy import signal
from transformers import AutoModel
import equalize_channel as EQ, equalize_v2_lib as E2
from sklearn.metrics import roc_curve, roc_auc_score
from cv_v2_probe import fit_score, eer
from src.data import read_audio_sf, crop_and_peak_normalize
SOS = signal.butter(4, [130.0, 3400.0], btype='bandpass', fs=EQ.SR, output='sos')
# gabarit LTAS identique a equalize_v2_full
bl = glob.glob('data_v2/real/bonafide/*.wav'); sl = [p for s in ('S01_xtts_wolof','S02_xtts_multi_ref','S03_speecht5_wolof','S05_knnvc_wolof') for p in glob.glob(f'data_v2/gen/{s}/*.wav')]
bl = sorted(bl, key=lambda p: os.path.basename(p)); 
bona = {os.path.basename(p)[:-4]: p for p in bl}
spoof = {}
for t, s in (('S01','S01_xtts_wolof'),('S02','S02_xtts_multi_ref'),('S03','S03_speecht5_wolof'),('S05','S05_knnvc_wolof')):
    for p in glob.glob(f'data_v2/gen/{s}/*.wav'):
        spoof.setdefault(re.sub(r'^S0\d_', '', os.path.basename(p)[:-4]), {})[t] = p
rng = random.Random(42); bl2 = list(dict(sorted(bona.items())).values()); sl2 = [p for u in sorted(spoof) for p in spoof[u].values()]
bl2 = list({os.path.basename(p)[:-4]: p for p in glob.glob('data_v2/real/bonafide/*.wav')}.values()); 
rng.shuffle(bl2); rng.shuffle(sl2)
target = EQ.compute_ltas_template(bl2[:80] + sl2[:80], SOS)
man = {r['utt_id']: r for r in json.load(open('data_v2/manifest_v2.json', encoding='utf-8'))['records']}
clips = []
for p in sorted(glob.glob('data_v2/pilot/spoof/S04_oolel_wolof/*.wav')):
    u = re.sub(r'^S04_', '', os.path.basename(p)[:-4]); x = EQ.read_wav_mono(p)[0]
    n = int(16000 * 4.1)
    if len(x) < n: print('trop court', u, len(x)/16000); continue
    seed = zlib.crc32(u.encode())
    y = E2.finalize(EQ.process_one(EQ.energy_crop_np(x, n), target, SOS, seed), target, SOS, seed ^ 0x5bd1e995)
    os.makedirs('data_v2/pilot_equalized/spoof/S04_oolel_wolof', exist_ok=True)
    out = f'data_v2/pilot_equalized/spoof/S04_oolel_wolof/S04_{u}.wav'; sf.write(out, y.astype(np.float32), 16000, subtype='PCM_16')
    clips.append((u, out, man[u]['fold'] if u in man else None))
print('clips S04 egalises', len(clips))
model = AutoModel.from_pretrained('pretrained_models/microsoft__wavlm-base').eval()
def feat(path):
    a = crop_and_peak_normalize(read_audio_sf(path), mode='center')
    with torch.no_grad():
        h = model(torch.from_numpy(a[None].astype(np.float32)), output_hidden_states=True).hidden_states[4][0]
    return np.concatenate([h.mean(0).numpy(), h.std(0).numpy()]).astype(np.float64)
d = 'experiments_pretrained/v2/wavlm'
idx = list(csv.DictReader(open(d + '/index.csv', encoding='utf-8')))
y = np.array([int(r['y']) for r in idx]); fold = np.array([int(r['fold']) for r in idx])
X = np.concatenate([np.load(d + '/mean.npy')[:, 4], np.load(d + '/std.npy')[:, 4]], 1).astype(np.float64)
oof = np.zeros(len(y))
for f in range(5):
    oof[fold == f] = fit_score(X[fold != f], y[fold != f], X[fold == f], 0.1)
fp, tp, th = roc_curve(y, oof); i = int(np.argmin(np.abs(fp - (1 - tp)))); thr = th[i]
print('seuil EER hors-pli %.3f (EER %.2f %%)' % (thr, eer(oof, y)))
sc = []
for u, p, f in clips:
    tr = fold != f
    sc.append(float(fit_score(X[tr], y[tr], feat(p)[None], 0.1)[0]))
sc = np.array(sc)
bsc = oof[y == 0]
print('scores S04', np.round(sc, 2).tolist())
print('detectes comme faux au seuil EER : %d / %d ; au seuil 0 : %d / %d' % ((sc > thr).sum(), len(sc), (sc > 0).sum(), len(sc)))
print('AUC S04 vs vrais (scores hors-pli) : %.3f' % roc_auc_score(np.r_[np.zeros(len(bsc)), np.ones(len(sc))], np.r_[bsc, sc]))
print('EER S04 vs vrais : %.2f %%' % eer(np.r_[bsc, sc], np.r_[np.zeros(len(bsc)), np.ones(len(sc))]))
