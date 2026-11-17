"""Extraction GELEE XLS-R 300M / MMS 300M sur la base preliminaire egalisee (condition `equalized`, 1 012 fichiers).
Meme recadrage central 4,0 s + pic que extract_pretrained.py ; normalisation zero-moyenne/variance unite comme extract_pretrained_v2.py.
Sorties experiments_pretrained/equalized/<modele>/ : index.csv (utt_id, split, spk, system, y), mean.npy, std.npy (N, 25, 1024), meta.json.
Usage : python extract_pretrained_xl_prelim.py [xlsr,mms]"""
import sys, json, time, csv
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from src.data import load_all_splits, read_audio_sf, crop_and_peak_normalize

torch.set_num_threads(6)
MODELS = {'xlsr': ('facebook__wav2vec2-xls-r-300m', True), 'mms': ('facebook__mms-300m', True)}
which = sys.argv[1] if len(sys.argv) > 1 else 'xlsr,mms'
sp = load_all_splits('equalized')
entries = [(s, e) for s in ('train', 'dev', 'eval') for e in sp[s]]
N = len(entries)
audio = np.stack([crop_and_peak_normalize(read_audio_sf(e.wav_path), mode='center') for _, e in entries])
print('audio', audio.shape, flush=True)
for key in which.split(','):
    name, do_norm = MODELS[key]
    out = Path(f'experiments_pretrained/equalized/{key}')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists():
        print(key, 'deja extrait', flush=True)
        continue
    model = AutoModel.from_pretrained(f'pretrained_models/{name}').eval()
    L, D = model.config.num_hidden_layers + 1, model.config.hidden_size
    x = audio.copy()
    if do_norm:
        x = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-7)
    mean = np.zeros((N, L, D), np.float32)
    std = np.zeros((N, L, D), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, 8):
            xb = torch.from_numpy(x[i:i + 8].astype(np.float32))
            H = torch.stack(model(xb, output_hidden_states=True).hidden_states, 1)
            mean[i:i + len(xb)] = H.mean(2).numpy()
            std[i:i + len(xb)] = H.std(2).numpy()
            if (i // 8) % 20 == 0:
                print(key, i, '/', N, '%.0fs' % (time.time() - t0), flush=True)
    np.save(out / 'mean.npy', mean)
    np.save(out / 'std.npy', std)
    with open(out / 'index.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['utt_id', 'split', 'spk', 'system', 'y'])
        for s, e in entries:
            w.writerow([e.utt_id, s, e.spk.upper(), e.system, e.label])
    json.dump(dict(model=name, n_files=N, n_layers_saved=L, crop_s=4.0, peak_normalized=True, model_normalize=do_norm,
                   seconds=round(time.time() - t0, 1), torch=torch.__version__), open(out / 'meta.json', 'w'), indent=1)
    del model
    print(key, 'termine', round(time.time() - t0), 's', flush=True)
