"""Egalisation de la base v2 complete : data_v2/real + data_v2/gen -> data_v2/eq (equalize_channel.process_one + equalize_v2_lib.finalize).
Duree identique au sein d'un groupe (meme texte source), gabarit LTAS commun calcule sur un echantillon de TOUTES les classes."""
import glob, os, re, random, zlib, json
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, soundfile as sf
from scipy import signal
import equalize_channel as EQ
import equalize_v2_lib as E2

SRC, DST = 'data_v2', 'data_v2/eq'
ATT = {'S01': 'S01_xtts_wolof', 'S02': 'S02_xtts_multi_ref', 'S03': 'S03_speecht5_wolof', 'S05': 'S05_knnvc_wolof'}
SOS = signal.butter(4, [130.0, 3400.0], btype='bandpass', fs=EQ.SR, output='sos')


def work(args):
    utt, items, target = args
    waves = [(t, EQ.read_wav_mono(src)[0], dst) for t, src, dst in items]
    n = min(len(x) for _, x, _ in waves)
    seed = zlib.crc32(utt.encode())
    for k, (t, x, dst) in enumerate(waves):
        fs = (seed + k * 9973) & 0xFFFFFFFF
        y = E2.finalize(EQ.process_one(EQ.energy_crop_np(x, n), target, SOS, fs), target, SOS, fs ^ 0x5bd1e995)
        sf.write(dst, y.astype(np.float32), EQ.SR, subtype='PCM_16')
    return utt, len(waves), n


if __name__ == '__main__':
    os.makedirs(f'{DST}/bonafide', exist_ok=True)
    for s in ATT.values():
        os.makedirs(f'{DST}/spoof/{s}', exist_ok=True)
    bona = {os.path.basename(p)[:-4]: p for p in glob.glob(f'{SRC}/real/bonafide/*.wav')}
    spoof = defaultdict(dict)
    for t, s in ATT.items():
        for p in glob.glob(f'{SRC}/gen/{s}/*.wav'):
            spoof[re.sub(r'^S0\d_', '', os.path.basename(p)[:-4])][t] = p
    rng = random.Random(42)
    bl = list(bona.values()); sl = [p for d in spoof.values() for p in d.values()]
    rng.shuffle(bl); rng.shuffle(sl)
    target = EQ.compute_ltas_template(bl[:80] + sl[:80], SOS)
    jobs = []
    for utt in sorted(bona):
        items = [('BONA', bona[utt], f'{DST}/bonafide/{utt}.wav')] + [(t, spoof[utt][t], f'{DST}/spoof/{ATT[t]}/{t}_{utt}.wav') for t in ATT if t in spoof[utt]]
        jobs.append((utt, items, target))
    with Pool(4) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            out.append(r)
            if i % 50 == 0:
                print('groupes', i + 1, '/', len(jobs), flush=True)
    json.dump(dict(groups=len(out), files=sum(o[1] for o in out), min_len_s=min(o[2] for o in out) / EQ.SR), open('data_v2/eq/eq_summary.json', 'w'))
    print('FINI groupes', len(out), 'fichiers', sum(o[1] for o in out), 'duree min %.2f s' % (min(o[2] for o in out) / EQ.SR))
