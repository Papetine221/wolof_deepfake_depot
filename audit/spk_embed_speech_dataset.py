"""Embeddings ECAPA-TDNN par (fichier, locuteur local) de speech_dataset (Kallaama wolof, probable).
Segments : tours MONO-locuteur (pas de chevauchement), >= 2 s, coupes a 8 s, au plus 10 par (fichier, locuteur), repartis dans le fichier.
Sorties : audit/spk_embed/entries.csv, emb_mean.npy (moyenne L2-normalisee), emb_halfA.npy / emb_halfB.npy (2 moities des segments,
servent a estimer la similarite 'meme personne'), seg_meta.csv. Ne modifie rien hors de audit/."""
import csv, glob, os, sys, xml.etree.ElementTree as ET
import numpy as np, soundfile as sf, torch
torch.set_num_threads(4)
from speechbrain.inference.speaker import EncoderClassifier

ROOT = 'speech_dataset'
OUT = 'audit/spk_embed'
os.makedirs(OUT, exist_ok=True)
MIN_S, MAX_S, NSEG = 2.0, 8.0, 10
clf = EncoderClassifier.from_hparams(source='pretrained_models/speechbrain__ecapa', savedir='pretrained_models/speechbrain__ecapa',
                                     run_opts={'device': 'cpu'})
entries, embs, hA, hB, segs = [], [], [], [], []
skipped = 0
files = sorted(glob.glob(f'{ROOT}/*/*.trs'), key=lambda p: (int(os.path.basename(os.path.dirname(p))), p))
for fi, trs in enumerate(files):
    wav = trs[:-4] + '.wav'
    info = sf.info(wav)
    sr = info.samplerate
    r = ET.parse(trs).getroot()
    by = {}
    for t in r.iter('Turn'):
        s = (t.get('speaker') or '').split()
        if len(s) != 1:
            continue  # chevauchement ou sans locuteur
        a, b = float(t.get('startTime')), float(t.get('endTime'))
        if b - a >= MIN_S:
            by.setdefault(s[0], []).append((a, min(b, a + MAX_S)))
    spk_meta = {s.get('id'): (s.get('type'), s.get('name')) for s in r.iter('Speaker')}
    for sid, turns in by.items():
        if len(turns) > NSEG:
            idx = np.linspace(0, len(turns) - 1, NSEG).round().astype(int)
            turns = [turns[i] for i in sorted(set(idx))]
        es = []
        for a, b in turns:
            x, _ = sf.read(wav, start=int(a * sr), stop=int(b * sr), dtype='float32', always_2d=True)
            x = x.mean(1)
            if len(x) < 1.0 * sr:  # tour dont les temps depassent la fin du fichier ou audio quasi vide
                skipped += 1
                continue
            if sr != 16000:
                import torchaudio
                x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
            with torch.no_grad():
                e = clf.encode_batch(torch.from_numpy(x)[None]).squeeze().numpy()
            es.append(e / np.linalg.norm(e))
            segs.append(dict(file=os.path.basename(wav)[:-4], dir=os.path.basename(os.path.dirname(wav)), spk=sid, start=a, end=b))
        if not es:
            continue
        turns = [(s_['start'], s_['end']) for s_ in segs[-len(es):]]
        E = np.stack(es)
        m = E.mean(0)
        entries.append(dict(dir=os.path.basename(os.path.dirname(wav)), file=os.path.basename(wav)[:-4], spk=sid, gender=spk_meta.get(sid, ('', ''))[0] or '',
                            n_seg=len(es), seconds=sum(b - a for a, b in turns)))
        embs.append(m / np.linalg.norm(m))
        if len(es) >= 4:
            ev, od = E[0::2].mean(0), E[1::2].mean(0)
            hA.append(ev / np.linalg.norm(ev)); hB.append(od / np.linalg.norm(od))
        else:
            hA.append(np.full(E.shape[1], np.nan)); hB.append(np.full(E.shape[1], np.nan))
    print(fi + 1, len(files), os.path.basename(wav), 'entrees', len(entries), flush=True)
np.save(f'{OUT}/emb_mean.npy', np.stack(embs)); np.save(f'{OUT}/emb_halfA.npy', np.stack(hA)); np.save(f'{OUT}/emb_halfB.npy', np.stack(hB))
for name, rows in (('entries', entries), ('seg_meta', segs)):
    with open(f'{OUT}/{name}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('FINI', len(entries), 'entrees', '| segments ignores (<1 s lus):', skipped)
