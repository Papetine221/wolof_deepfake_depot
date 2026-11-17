"""LCNN-Mel (hyperparametres INCHANGES de train_all.train_single_run) sur la base v2 egalisee, memes protocoles que cv_v2_probe.py.
Pli f : test = pli f ; dev = pli (f+1)%5 (choix de l'epoque) ; train = 3 autres plis. Protocole 'logo' : la famille laissee de cote est retiree du train ET du dev ;
test = vrais du pli + famille laissee de cote. Standardisation par bande sur le train du pli.
Usage : python cv_v2_lcnn.py plis | logo      Sorties : experiments_pretrained/v2/lcnn/<proto>/<fam>/fold<k>/ , cv_summary_<proto>.json"""
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.')
import train_all
from src.data import ProtocolEntry, read_audio_sf
from src.features import extract_feature_from_waveform, apply_std_floor, CROP_FRAMES
torch.set_num_threads(6)
proto = sys.argv[1]
FAM = {'XTTS': ('S01', 'S02'), 'SpeechT5': ('S03',), 'kNNVC': ('S05',)}
OUT = Path('experiments_pretrained/v2/lcnn'); CACHE = OUT / 'cache_mel'; CACHE.mkdir(parents=True, exist_ok=True)
idx = list(csv.DictReader(open('experiments_pretrained/v2/wavlm/index.csv', encoding='utf-8')))
ents, cache = [], {}
for r in idx:
    sy = r['system']
    e = ProtocolEntry(spk=r['person'], utt_id=r['utt_id'], system='-' if sy == 'bonafide' else sy, label_str='bonafide' if sy == 'bonafide' else 'spoof',
                      label=int(r['y']), wav_path=Path(r['path']))
    ents.append((e, int(r['fold'])))
    p = CACHE / f'{r["utt_id"]}.npy'
    if p.exists():
        cache[r['utt_id']] = np.load(p)
    else:
        cache[r['utt_id']] = extract_feature_from_waveform(read_audio_sf(Path(r['path'])), 'mel'); np.save(p, cache[r['utt_id']])
print('features', len(cache), flush=True)
fams = [('tous', ())] if proto == 'plis' else list(FAM.items())
summary = {}
for fam, drop in fams:
    for f in range(5):
        dv = (f + 1) % 5
        keep = lambda e: e.system not in drop
        tr = [e for e, k in ents if k not in (f, dv) and keep(e)]
        de = [e for e, k in ents if k == dv and keep(e)]
        te = [e for e, k in ents if k == f and (not drop or e.system == '-' or e.system in drop)]
        d = OUT / proto / fam / f'fold{f}'
        if not (d / 'metrics.json').exists():
            sl = []
            for e in tr:
                fm = cache[e.utt_id]; s0 = max(0, (fm.shape[1] - CROP_FRAMES) // 2); sl.append(fm[:, s0:s0 + CROP_FRAMES])
            cat = np.concatenate(sl, 1); mean = cat.mean(1).astype(np.float32)
            std, info = apply_std_floor(np.maximum(cat.std(1), 1e-6), 'relative')
            t0 = time.time()
            m = train_all.train_single_run('v2eq', 'mel', 0, dict(train=tr, dev=de, eval=te), cache, mean, std, d, epochs=30, std_info=info)
            print(proto, fam, 'pli', f, 'n', len(tr), len(de), len(te), '%.0fs' % (time.time() - t0), 'dev EER %.2f TEST EER %.2f AUC %.4f' % (m['dev_eer'], m['eval_eer'], m['eval_auc']), m['eer_per_system'], flush=True)
        mm = json.load(open(d / 'metrics.json'))
        summary[f'{fam}|fold{f}'] = dict(test_eer=mm['eval_eer'], test_auc=mm['eval_auc'], dev_eer=mm['dev_eer'], eer_per_system=mm['eer_per_system'], best_epoch=mm['best_epoch'])
        json.dump(summary, open(OUT / f'cv_summary_{proto}.json', 'w'), indent=1)
for fam, _ in fams:
    v = [summary[f'{fam}|fold{f}']['test_eer'] for f in range(5)]
    print('RESUME', proto, fam, 'EER moyen %.2f +- %.2f' % (np.mean(v), np.std(v, ddof=1)), [round(x, 1) for x in v])
