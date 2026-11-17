"""Plan fige de la base v2 (A2-A4) : selection des personnes/clips, export des vrais clips et des references, manifeste.
Choix (voir audit/PLAN_base_v2.md) : segments 4,0-8,0 s, francais <= 10 %, recadrage 4,5 s (acceptation >= 4,1 s ; les modeles aval recadrent a 4,0 s) pour tous (vrais et faux), au plus 10 clips par personne,
au moins 5, 1 a 3 segments de reference par personne (concatenes, hors clips), plis = groupes de personnes (protocols_groups/folds_t050.json),
cibles S02 et S05 = AUTRES personnes du MEME pli. Ecrit uniquement dans data_v2/real/ et data_v2/manifest_v2.json."""
import csv, json, os, random, collections
import numpy as np, soundfile as sf, torch
import generation_faux as G

CL = G.Config
CL.MIN_SEG_DUR, CL.MAX_SEG_DUR, CL.MAX_FRENCH_RATIO = 4.0, 8.0, 0.10
CL.TARGET_LEN = int(16000 * 4.5)
CL.MIN_SAMPLES = int(16000 * 4.1)
ROOT = 'data_v2/real'
CL.BONAFIDE, CL.REF_DIR = f'{ROOT}/bonafide', f'{ROOT}/refs'
os.makedirs(CL.BONAFIDE, exist_ok=True); os.makedirs(CL.REF_DIR, exist_ok=True)
KMAX, KMIN = 10, 5

persons = {(r['file'], r['spk']): r['person'] for r in csv.DictReader(open('protocols_groups/persons_t050.csv', encoding='utf-8'))}
fold_of = {p: int(k) for k, f in json.load(open('protocols_groups/folds_t050.json'))['folds'].items() for p in f['person_groups']}
segs = G.parse_all_segments()
by = collections.defaultdict(list)
for s in segs:
    p = persons.get((s['file_id'], s['locuteur'].replace('locuteur_', 'spk')))
    if p:
        by[p].append(s)
rng = random.Random(42)
recs, kept = [], {}
for p in sorted(by):
    pool = sorted(by[p], key=lambda s: (s['file_id'], s['start']))
    rng.shuffle(pool)
    nref = min(3, max(1, len(pool) - KMIN))
    ncont = min(KMAX, len(pool) - nref)
    if ncont < KMIN:
        continue
    gender = max(('female', 'male'), key=lambda g: sum(1 for s in pool if s['gender'] == g))
    kept[p] = dict(fold=fold_of[p], gender=gender, n_available=len(pool))
    for j, s in enumerate(pool[nref:nref + ncont]):
        r = dict(s); r.update(person=p, spk_id=p, fold=fold_of[p], person_gender=gender, utt_id=f'{p.lower()}_{j:03d}', is_ref=False, text_tts=G.clip_text(s['text']))
        recs.append(r)
    for j, s in enumerate(pool[:nref]):
        r = dict(s); r.update(person=p, spk_id=p, fold=fold_of[p], person_gender=gender, utt_id=f'{p.lower()}_ref{j}', is_ref=True, text_tts=G.clip_text(s['text']))
        recs.append(r)
# cibles S02 / S05 : autres personnes du MEME pli (rotation deterministe)
pf = collections.defaultdict(list)
for p, d in kept.items():
    pf[d['fold']].append(p)
for k in pf:
    pf[k].sort()
bad = [k for k, v in pf.items() if len(v) < 3]
assert not bad, f'plis avec < 3 personnes : {bad}'
rot2, rot5 = collections.Counter(), collections.Counter()
for r in recs:
    if r['is_ref']:
        continue
    others = [q for q in pf[r['fold']] if q != r['person']]
    r['ref_s01'] = f"{r['person'].lower()}_ref"
    r['ref_s02_person'] = others[rot2[r['person']] % len(others)]
    r['ref_s05_person'] = others[(rot5[r['person']] + 1 + len(others) // 2) % len(others)]
    r['ref_s02'] = f"{r['ref_s02_person'].lower()}_ref"
    r['ref_s05'] = f"{r['ref_s05_person'].lower()}_ref"
    rot2[r['person']] += 1; rot5[r['person']] += 1
# export : vrais clips = chaine commune (recadrage 4 s + canal + niveau) ; refs = concatenation des segments de reference, 16 kHz, niveau normalise, SANS recadrage
exp = []
for r in recs:
    if r['is_ref']:
        continue
    wav, sr = G.load_slice(r['src_wav'], r['start'], r['end'])
    if wav is None:
        continue
    out = f"{CL.BONAFIDE}/{r['utt_id']}.wav"
    s = G.postprocess_and_save(wav, sr, out)
    if s is not None:
        r['wav_path'] = out.replace(chr(92), '/'); r['n_samples'] = int(s.numel()); exp.append(r)
for p in kept:
    parts = []
    for r in [r for r in recs if r['is_ref'] and r['person'] == p]:
        w, sr = G.load_slice(r['src_wav'], r['start'], r['end'])
        if w is not None:
            parts.append(G.peak_norm(G.to_mono_16k(w, sr)))
    ref = torch.cat(parts) if parts else None
    if ref is not None:
        sf.write(f"{CL.REF_DIR}/{p.lower()}_ref.wav", ref.numpy(), 16000, subtype='PCM_16')
        kept[p]['ref_seconds'] = round(ref.numel() / 16000, 1)
have = {r['person'] for r in exp}
exp = [r for r in exp if os.path.exists(f"{CL.REF_DIR}/{r['person'].lower()}_ref.wav") and r['ref_s02_person'] in have and r['ref_s05_person'] in have]
json.dump(dict(config=dict(seg_dur=[4.0, 8.0], french_max=0.10, crop_s=4.5, kmax=KMAX, kmin=KMIN, threshold_persons=0.5), persons=kept, records=exp), open('data_v2/manifest_v2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
fc = collections.Counter(r['fold'] for r in exp); pc = collections.Counter(d['fold'] for d in kept.values())
print('personnes', len(kept), 'clips reels exportes', len(exp), '| par pli : clips', dict(sorted(fc.items())), 'personnes', dict(sorted(pc.items())))
print('femmes (majorite) :', sum(1 for d in kept.values() if d['gender'] == 'female'), '| refs (s) mediane', np.median([d.get('ref_seconds', 0) for d in kept.values()]))
