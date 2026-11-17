"""Extraction GELEE wav2vec2-base / WavLM-base sur la base v2 egalisee (data_v2/eq) : recadrage central 4,0 s + pic, comme extract_pretrained.py.
Sorties experiments_pretrained/v2/<modele>/ : index.csv (utt_id, system, person, fold, y, path), mean.npy, std.npy (N,13,768), meta.json."""
import sys, json, time, csv, glob, re, os
from pathlib import Path
import numpy as np, torch
from transformers import AutoModel
from src.data import read_audio_sf, crop_and_peak_normalize
torch.set_num_threads(6)
MODELS = {'wav2vec2': ('facebook__wav2vec2-base', True), 'wavlm': ('microsoft__wavlm-base', False),
          'xlsr': ('facebook__wav2vec2-xls-r-300m', True), 'mms': ('facebook__mms-300m', True)}
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
man = {r['utt_id']: r for r in json.load(open('data_v2/manifest_v2.json', encoding='utf-8'))['records']}
rows = []
for pat, sy in [('data_v2/eq/bonafide/*.wav', 'bonafide')] + [(f'data_v2/eq/spoof/{s}*/*.wav', s[:3]) for s in ('S01', 'S02', 'S03', 'S05')]:
    for p in sorted(glob.glob(pat)):
        u = re.sub(r'^S0\d_', '', os.path.basename(p)[:-4]); r = man[u]
        rows.append(dict(utt_id=os.path.basename(p)[:-4], system=sy, person=r['person'], fold=r['fold'], y=int(sy != 'bonafide'), src=u, path=p.replace(chr(92), '/')))
N = len(rows)
audio = np.stack([crop_and_peak_normalize(read_audio_sf(r['path']), mode='center') for r in rows])
print('audio', audio.shape, flush=True)
for key in (['wav2vec2', 'wavlm'] if which == 'both' else which.split(',')):
    name, do_norm = MODELS[key]
    out = Path(f'experiments_pretrained/v2/{key}'); out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists():
        print(key, 'deja extrait', flush=True); continue
    model = AutoModel.from_pretrained(f'pretrained_models/{name}').eval()
    L = model.config.num_hidden_layers + 1
    x = audio.copy()
    if do_norm:
        x = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-7)
    D = model.config.hidden_size
    mean = np.zeros((N, L, D), np.float32); std = np.zeros((N, L, D), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, 8):
            xb = torch.from_numpy(x[i:i + 8].astype(np.float32))
            H = torch.stack(model(xb, output_hidden_states=True).hidden_states, 1)
            mean[i:i + len(xb)] = H.mean(2).numpy(); std[i:i + len(xb)] = H.std(2).numpy()
            if (i // 8) % 40 == 0:
                print(key, i, '/', N, '%.0fs' % (time.time() - t0), flush=True)
    np.save(out / 'mean.npy', mean); np.save(out / 'std.npy', std)
    with open(out / 'index.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(model=name, n_files=N, crop_s=4.0, seconds=round(time.time() - t0, 1), torch=torch.__version__), open(out / 'meta.json', 'w'), indent=1)
    print(key, 'termine', round(time.time() - t0), 's', flush=True)
