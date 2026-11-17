"""Generation des faux de la base v2 (S01/S02 XTTS wolof, S03 SpeechT5, S05 kNN-VC) a partir de data_v2/manifest_v2.json. Reprenable (fichiers existants ignores),
avec --limit (essai) et --shard i/n. Utilise les fonctions de generation_faux.py inchangees. Fonctionne en local (CPU) et sur Colab (GPU).
Usage : python generation_v2_full.py s01s02|s03|s05 [--limit N] [--shard i/n]
Sorties : data_v2/gen/S0x_*/S0x_<utt_id>.wav  (puis egalisation et controles en local)."""
import argparse, json, os, sys
import generation_faux as G

CL = G.Config
CL.TARGET_LEN = int(16000 * 4.5)
CL.MIN_SAMPLES = int(16000 * 4.1)
RT = os.environ.get('V2_ROOT', 'data_v2')
CL.BONAFIDE, CL.REF_DIR = f'{RT}/real/bonafide', f'{RT}/real/refs'
CL.S01_DIR, CL.S02_DIR, CL.S03_DIR = f'{RT}/gen/S01_xtts_wolof', f'{RT}/gen/S02_xtts_multi_ref', f'{RT}/gen/S03_speecht5_wolof'
S05_DIR = f'{RT}/gen/S05_knnvc_wolof'
for d in (CL.S01_DIR, CL.S02_DIR, CL.S03_DIR, S05_DIR):
    os.makedirs(d, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument('step'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', default='0/1')
a = ap.parse_args()
recs = json.load(open(f'{RT}/manifest_v2.json', encoding='utf-8'))['records']
i, n = map(int, a.shard.split('/'))
recs = [r for k, r in enumerate(sorted(recs, key=lambda r: r['utt_id'])) if k % n == i]
if a.limit:
    recs = recs[:a.limit]
print('etape', a.step, '| clips', len(recs), '| dispositif', CL.DEVICE, flush=True)


def s05(recs):
    import torch
    local = 'pretrained_models/knn_vc'
    if os.path.isdir(f'{local}/code'):  # local : poids deja telecharges
        sys.path.insert(0, f'{local}/code')
        def _l(url, map_location=None, progress=True, **kw):
            return torch.load(f"{local}/" + url.rsplit('/', 1)[-1], map_location=map_location, weights_only=False)
        torch.hub.load_state_dict_from_url = _l
        import hubconf
        vc = hubconf.knn_vc(pretrained=True, prematched=True, device=CL.DEVICE)
    else:  # Colab : torch.hub telecharge le code et les poids (MIT)
        vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=CL.DEVICE)
    by = {}
    for r in json.load(open(f'{RT}/manifest_v2.json', encoding='utf-8'))['records']:
        by.setdefault(r['person'], []).append(r)
    pool = {}
    def mset(p):
        if p not in pool:
            ws = [f'{CL.REF_DIR}/{p.lower()}_ref.wav'] + [f'{CL.BONAFIDE}/{r["utt_id"]}.wav' for r in by[p]]
            pool[p] = vc.get_matching_set(ws)
        return pool[p]
    ok = 0
    for r in recs:
        out = f"{S05_DIR}/S05_{r['utt_id']}.wav"
        if os.path.exists(out):
            continue
        try:
            q = vc.get_features(f"{CL.BONAFIDE}/{r['utt_id']}.wav")
            wav = vc.match(q, mset(r['ref_s05_person']), topk=4)
            ok += G.postprocess_and_save(wav.cpu().numpy(), 16000, out) is not None
        except Exception as e:
            print('ECHEC S05', r['utt_id'], str(e)[:150], flush=True)
    print('S05 ecrits', ok)


if a.step == 's01s02':
    model, _ = G.load_xtts_model()
    G.generate_s01_s02(model, recs)
elif a.step == 's03':
    G.generate_s03(recs)
elif a.step == 's05':
    s05(recs)
else:
    sys.exit('etape inconnue')
