"""Etape 2 : extraction GELEE (eval, no_grad) des representations wav2vec2-base / WavLM-base sur la condition `equalized`.
Meme recadrage central 4,0 s + normalisation par le pic que le LCNN (src.data), puis normalisation propre au modele
(zero-moyenne/variance unite pour wav2vec2 ; aucune pour WavLM, selon preprocessor_config.json).
Sorties (experiments_pretrained/equalized/<modele>/) :
  index.csv                 utt_id, split, spk, system, y  (ordre = lignes des tableaux)
  mean.npy, std.npy         (N, 13, 768) float32 : moyenne / ecart-type temporels par couche (0 = sortie des convolutions projetee)
  layer8_hidden.npy         (N, T, 768) float16 : sortie de la couche 8 (pour re-entrainer les 4 dernieres couches, etape 4)
  meta.json                 revision, parametres, durees
Usage : python extract_pretrained.py [wav2vec2|wavlm|both]"""
import sys, json, time, csv
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from src.data import load_all_splits, read_audio_sf, crop_and_peak_normalize

torch.set_num_threads(4)
MODELS = {'wav2vec2': ('facebook__wav2vec2-base', True), 'wavlm': ('microsoft__wavlm-base', False)}
BATCH = 8
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
sp = load_all_splits('equalized')
entries = [(s, e) for s in ('train', 'dev', 'eval') for e in sp[s]]
N = len(entries)
audio = np.stack([crop_and_peak_normalize(read_audio_sf(e.wav_path), mode='center') for _, e in entries])  # (N, 64000)
print('audio', audio.shape, 'N', N, flush=True)

for key in (['wav2vec2', 'wavlm'] if which == 'both' else [which]):
    name, do_norm = MODELS[key]
    out = Path(f'experiments_pretrained/equalized/{key}')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists():
        print(key, 'deja extrait, saut', flush=True)
        continue
    model = AutoModel.from_pretrained(f'pretrained_models/{name}').eval()
    L = model.config.num_hidden_layers + 1
    x = audio.copy()
    if do_norm:
        x = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-7)
    mean = np.zeros((N, L, 768), np.float32)
    std = np.zeros((N, L, 768), np.float32)
    layer8 = None
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, BATCH):
            xb = torch.from_numpy(x[i:i + BATCH].astype(np.float32))
            hs = model(xb, output_hidden_states=True).hidden_states  # L x (B, T, 768)
            H = torch.stack(hs, 1)  # (B, L, T, 768)
            mean[i:i + len(xb)] = H.mean(2).numpy()
            std[i:i + len(xb)] = H.std(2).numpy()
            if layer8 is None:
                layer8 = np.lib.format.open_memmap(out / 'layer8_hidden.npy', mode='w+', dtype=np.float16, shape=(N, H.shape[2], 768))
            layer8[i:i + len(xb)] = H[:, 8].numpy().astype(np.float16)
            if (i // BATCH) % 20 == 0:
                print(key, i, '/', N, '%.0fs' % (time.time() - t0), flush=True)
    layer8.flush()
    np.save(out / 'mean.npy', mean)
    np.save(out / 'std.npy', std)
    with open(out / 'index.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['utt_id', 'split', 'spk', 'system', 'y'])
        for s, e in entries:
            w.writerow([e.utt_id, s, e.spk.upper(), e.system, e.label])
    json.dump(dict(model=name, revision={'facebook__wav2vec2-base': '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8',
                                         'microsoft__wavlm-base': 'efa81aae7ff777e464159e0f877d54eac5b84f81'}[name],
                   n_files=N, n_layers_saved=L, crop_s=4.0, peak_normalized=True, model_normalize=do_norm,
                   frames=int(layer8.shape[1]), seconds=round(time.time() - t0, 1), torch=torch.__version__), open(out / 'meta.json', 'w'), indent=1)
    print(key, 'termine', round(time.time() - t0, 1), 's', flush=True)
