import os

if not os.environ.get("HF_ONLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import gc
import re
import sys
import glob
import json
import math
import zlib
import random
import zipfile
import logging
import argparse
import subprocess
from collections import defaultdict

import torch
import torchaudio
import torchaudio.functional as AF
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from huggingface_hub import hf_hub_download, login
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

STM_LINE_RE = re.compile(
    r"^(\S+)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+<([^>]*)>\s*(.*)$"
)


class Config:
    BASE_DIR = "data"
    SPEECH_DATASET_DIR = "speech_dataset"
    BONAFIDE_OLD = f"{BASE_DIR}/bonafide"
    BONAFIDE = f"{BASE_DIR}/bonafide_new"
    REF_DIR = f"{BASE_DIR}/refs_new"
    SPOOF_DIR = f"{BASE_DIR}/spoof_new"
    MODEL_DIR = f"{BASE_DIR}/models/xtts_wolof"
    PROTO_DIR = f"{BASE_DIR}/protocols_new"

    S01_DIR = f"{SPOOF_DIR}/S01_xtts_wolof"
    S02_DIR = f"{SPOOF_DIR}/S02_xtts_multi_ref"
    S03_DIR = f"{SPOOF_DIR}/S03_speecht5_wolof"

    XTTS_ROOT = f"{MODEL_DIR}/galsenai-xtts-wo-checkpoints"
    XTTS_CHECKPOINT_DIR = f"{XTTS_ROOT}/Anta_GPT_XTTS_Wo"
    XTTS_CHECKPOINT = f"{XTTS_CHECKPOINT_DIR}/best_model_89250.pth"
    XTTS_CONFIG = f"{XTTS_CHECKPOINT_DIR}/config.json"
    XTTS_VOCAB = f"{XTTS_ROOT}/XTTS_v2.0_original_model_files/vocab.json"

    TARGET_SR = 16000
    TARGET_LEN = 16000 * 5
    MIN_SAMPLES = int(16000 * 4.5)
    XTTS_SR = 24000
    XTTS_CHAR_LIMIT = 250

    N_SHOWS = 10
    MAX_SPK_PER_SHOW = 3
    N_SPEAKERS = 10
    CLIPS_PER_SPEAKER = 30
    CLIP_MARGIN = 2  # 32 clips/locuteur (30 + 2 de marge pour absorber les spoofs perdus)
    CPU_THREADS = 4  # optimal sur 4 cœurs physiques MKL
    MIN_SEG_DUR = 4.5
    MAX_SEG_DUR = 6.5
    MIN_TEXT_CHARS = 12
    MIN_TEXT_WORDS = 3
    MAX_FRENCH_RATIO = 0.10  # part max de mots français par segment

    # Chaîne de dégradation canal commune bona fide / spoof (anti-raccourci)
    CHANNEL_MATCH = True
    CHANNEL_HP_HZ = 80
    CHANNEL_LP_HZ = 7000
    CHANNEL_SNR_DB = (20.0, 35.0)

    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    XTTS_REPO = "galsenai/xTTS-v2-wolof"
    SPEECHT5_REPO = "bilalfaye/speecht5_tts-wolof"
    SPEECHT5_VOCODER = "microsoft/speecht5_hifigan"
    SPEECHT5_XVECTORS = "Matthijs/cmu-arctic-xvectors"
    SPEECHT5_XVECTOR_IDX = [7306, 1138, 2300, 3500, 4700, 5900, 6800, 7600, 800, 1800]

    # Code TTS local (Coqui fork Wolof-TTS). Surcharger sur Colab via WOLOF_TTS_ROOT.
    WOLOF_TTS_ROOT = os.environ.get(
        "WOLOF_TTS_ROOT",
        os.path.join("Wolof-TTS", "notebooks", "Models", "xTTS v2"),
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MANIFEST_PATH = os.path.join(Config.BASE_DIR, "manifest_stm.json")
METADATA_PATH = os.path.join(Config.BASE_DIR, "dataset_metadata_new.json")
QC_PATH = os.path.join(Config.BASE_DIR, "qc_channel_shortcut.json")
FOLDS_PATH = os.path.join(Config.PROTO_DIR, "speaker_folds.json")


def setup_directories():
    for d in [
        Config.BONAFIDE,
        Config.REF_DIR,
        Config.SPOOF_DIR,
        Config.MODEL_DIR,
        Config.PROTO_DIR,
        Config.S01_DIR,
        Config.S02_DIR,
        Config.S03_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
    logging.info("Dossiers de sortie (bonafide_new / spoof_new) créés.")


def clean_stm_text(text):
    text = text or ""
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"%\S+", " ", text)
    text = re.sub(r":[A-Za-z]{2,}\b", " ", text)
    text = text.replace("ignore_time_segment_in_scoring", " ")
    text = re.sub(r"[<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


FRENCH_TOKENS = {
    "voilà", "voila", "merci", "accord", "vrai", "vraiment", "que", "qui", "est", "les",
    "pour", "avec", "mais", "donc", "alors", "bonjour", "bonsoir", "très", "aussi",
    "parce", "cette",
}
FRENCH_ELISION_RE = re.compile(r"\b[cdjlnqst]['’]\s?\w", re.IGNORECASE)


def french_ratio(raw_text, cleaned):
    """Part de mots français : balises :fra du STM, lexique de mots-outils, élisions."""
    n_words = max(1, len(cleaned.split()))
    n_tag = len(re.findall(r"\S+:fra\b", raw_text or ""))
    low = cleaned.lower()
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", low)
    n_lex = sum(1 for w in words if w in FRENCH_TOKENS)
    n_el = len(FRENCH_ELISION_RE.findall(low))
    return (max(n_tag, n_lex) + n_el) / n_words


def is_french_codeswitch(raw_text, cleaned):
    """Rejette les segments trop français (> MAX_FRENCH_RATIO des mots). Le wolof radio
    mélange beaucoup de français : un rejet strict (:fra) éliminerait ~83 % du corpus."""
    return french_ratio(raw_text, cleaned) > Config.MAX_FRENCH_RATIO

def clip_text(text, limit=Config.XTTS_CHAR_LIMIT):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip()


def parse_stm_gender(tag):
    tag = (tag or "").lower()
    if "female" in tag:
        return "female"
    if "male" in tag:
        return "male"
    return "unknown"


def find_wav_for_stm(show_id, file_id):
    direct = os.path.join(Config.SPEECH_DATASET_DIR, show_id, f"{file_id}.wav")
    if os.path.isfile(direct):
        return direct
    matches = glob.glob(os.path.join(Config.SPEECH_DATASET_DIR, show_id, f"{file_id}.*"))
    matches = [m for m in matches if m.lower().endswith((".wav", ".flac", ".mp3"))]
    return matches[0] if matches else None


def iter_stm_paths():
    """Yield (show_id, stm_path, checked). Checked transcriptions override."""
    checked_root = os.path.join(Config.SPEECH_DATASET_DIR, "checked_transcriptions")
    seen = set()
    if os.path.isdir(checked_root):
        for stm in glob.glob(os.path.join(checked_root, "*", "stm_format", "*.stm")):
            show_id = os.path.basename(os.path.dirname(os.path.dirname(stm)))
            file_id = os.path.splitext(os.path.basename(stm))[0]
            seen.add((show_id, file_id))
            yield show_id, stm, True
    for stm in glob.glob(os.path.join(Config.SPEECH_DATASET_DIR, "*", "stm_format", "*.stm")):
        show_id = os.path.basename(os.path.dirname(os.path.dirname(stm)))
        if show_id == "checked_transcriptions":
            continue
        file_id = os.path.splitext(os.path.basename(stm))[0]
        if (show_id, file_id) in seen:
            continue
        yield show_id, stm, False


def parse_stm_file(show_id, stm_path, checked):
    file_id = os.path.splitext(os.path.basename(stm_path))[0]
    wav_path = find_wav_for_stm(show_id, file_id)
    if not wav_path:
        logging.warning(f"WAV introuvable pour {show_id}/{file_id}")
        return []

    segments = []
    with open(stm_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            m = STM_LINE_RE.match(line)
            if not m:
                continue
            _fid, _ch, locuteur, t0, t1, tag, raw_text = m.groups()
            locuteur = locuteur.strip()
            if locuteur in {"excluded_region", "inter_segment_gap"}:
                continue
            if "ignore_time_segment_in_scoring" in raw_text:
                continue
            start = float(t0)
            end = float(t1)
            dur = end - start
            if dur < Config.MIN_SEG_DUR or dur > Config.MAX_SEG_DUR:
                continue
            text = clean_stm_text(raw_text)
            if is_french_codeswitch(raw_text, text):
                continue
            if len(text) < Config.MIN_TEXT_CHARS:
                continue
            if len(text.split()) < Config.MIN_TEXT_WORDS:
                continue
            lowered = text.lower()
            if lowered.replace("hum", "").strip() in {"", "ah", "oh"}:
                continue
            segments.append(
                {
                    "show_id": show_id,
                    "file_id": file_id,
                    "src_wav": wav_path.replace("\\", "/"),
                    "stm_path": stm_path.replace("\\", "/"),
                    "checked": checked,
                    "locuteur": locuteur,
                    "gender": parse_stm_gender(tag),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(dur, 3),
                    "text": text,
                    "text_tts": clip_text(text),
                    "raw_spk": f"{show_id}_{locuteur}",
                }
            )
    return segments


def parse_all_segments():
    all_segs = []
    n_stm = 0
    n_checked = 0
    for show_id, stm_path, checked in iter_stm_paths():
        n_stm += 1
        if checked:
            n_checked += 1
        all_segs.extend(parse_stm_file(show_id, stm_path, checked))
    logging.info(
        f"{len(all_segs)} segments STM valides "
        f"({Config.MIN_SEG_DUR}-{Config.MAX_SEG_DUR}s, sans code-switching français) "
        f"depuis {n_stm} fichiers STM ({n_checked} checked)."
    )
    return all_segs


def select_minimal_utterances(segments):
    by_spk = defaultdict(list)
    for seg in segments:
        by_spk[seg["raw_spk"]].append(seg)

    need = Config.CLIPS_PER_SPEAKER + Config.CLIP_MARGIN + 1
    ranked = []
    for raw_spk, segs in by_spk.items():
        segs = sorted(segs, key=lambda s: (s["duration"], s["start"]))
        n_checked = sum(1 for s in segs if s["checked"])
        shows = {s["show_id"] for s in segs}
        genders = {s["gender"] for s in segs if s["gender"] != "unknown"}
        gender = next(iter(genders), "unknown")
        ranked.append(
            {
                "raw_spk": raw_spk,
                "segs": segs,
                "n": len(segs),
                "n_checked": n_checked,
                "shows": shows,
                "gender": gender,
            }
        )

    ranked.sort(key=lambda r: (-r["n_checked"], -r["n"], r["raw_spk"]))
    eligible = [r for r in ranked if r["n"] >= need]
    if len(eligible) < Config.N_SPEAKERS:
        logging.warning(
            f"Seulement {len(eligible)} locuteurs avec {need} segments. "
            "Assouplissement du quota de clips."
        )
        eligible = [r for r in ranked if r["n"] >= 8]
        if not eligible:
            raise RuntimeError("Pas assez de locuteurs STM exploitables.")

    selected = []
    show_count = defaultdict(int)
    gender_counts = {"female": 0, "male": 0, "unknown": 0}

    def allowed(row):
        if any(show_count.get(sh, 0) >= Config.MAX_SPK_PER_SHOW for sh in row["shows"]):
            return False
        new = [sh for sh in row["shows"] if sh not in show_count]
        return len(show_count) + len(new) <= Config.N_SHOWS

    def take(row):
        selected.append(row)
        for sh in row["shows"]:
            show_count[sh] += 1
        gender_counts[row["gender"]] = gender_counts.get(row["gender"], 0) + 1

    # 1) femmes (~1/3 des places), émission nouvelle de préférence
    n_f = max(1, Config.N_SPEAKERS // 3)
    for row in eligible:
        if gender_counts["female"] >= n_f or len(selected) >= Config.N_SPEAKERS:
            break
        if row["gender"] == "female" and allowed(row) and not (row["shows"] & set(show_count)):
            take(row)
    # 2) émissions nouvelles, 3) complétion avec plafond MAX_SPK_PER_SHOW
    for new_only in (True, False):
        for row in eligible:
            if len(selected) >= Config.N_SPEAKERS:
                break
            if row in selected or not allowed(row):
                continue
            if new_only and (row["shows"] & set(show_count)):
                continue
            take(row)
    selected = selected[: Config.N_SPEAKERS]
    logging.info(
        f"Émissions retenues : {len(show_count)} (max {Config.MAX_SPK_PER_SHOW} locuteurs/émission)."
    )

    clips_per = min(
        Config.CLIPS_PER_SPEAKER + Config.CLIP_MARGIN, min(r["n"] - 1 for r in selected)
    )
    if clips_per < Config.CLIPS_PER_SPEAKER:
        logging.warning(f"CLIPS_PER_SPEAKER réduit à {clips_per}.")

    rng = random.Random(42)
    records = []
    speaker_map = {}
    for i, row in enumerate(selected):
        spk_id = f"SPK{i:02d}"
        speaker_map[spk_id] = {
            "raw_spk": row["raw_spk"],
            "gender": row["gender"],
            "shows": sorted(row["shows"]),
            "n_available": row["n"],
        }
        pool = list(row["segs"])
        rng.shuffle(pool)  # échantillon aléatoire (pas de biais vers les segments courts)
        # 1 ref (hors détection) puis contenus, tirés au hasard (seed 42)
        ref_seg = pool[0]
        content = pool[1 : 1 + clips_per]
        for j, seg in enumerate(content):
            rec = dict(seg)
            rec["spk_id"] = spk_id
            rec["utt_id"] = f"spk{i:02d}_{j:03d}"
            rec["is_ref"] = False
            rec["text_tts"] = clip_text(seg["text"])
            records.append(rec)
        ref = dict(ref_seg)
        ref["spk_id"] = spk_id
        ref["utt_id"] = f"spk{i:02d}_ref"
        ref["is_ref"] = True
        ref["text_tts"] = clip_text(ref_seg["text"])
        records.append(ref)

    shows = sorted({s["show_id"] for s in records})
    logging.info(
        f"Sélection : {len(selected)} locuteurs, {len(shows)} émissions, "
        f"{sum(1 for r in records if not r['is_ref'])} bona fide + "
        f"{sum(1 for r in records if r['is_ref'])} refs. "
        f"Genre {gender_counts}."
    )
    return records, speaker_map, clips_per


def assign_splits_and_refs(records, speaker_map):
    """Splits disjoints par locuteur ET par émission (pas de fuite canal/studio)."""
    spk_show = {spk: sorted(info["shows"])[0] for spk, info in speaker_map.items()}
    show_spks = defaultdict(list)
    for spk, sh in spk_show.items():
        show_spks[sh].append(spk)
    shows = sorted(show_spks)
    random.Random(42).shuffle(shows)
    n = len(spk_show)
    n_eval = n_dev = max(2, int(round(n * 0.2))) if n >= 6 else 1

    split_of = {}
    counts = {"eval": 0, "dev": 0, "train": 0}
    for sh in shows:
        if counts["eval"] < n_eval:
            sp = "eval"
        elif counts["dev"] < n_dev:
            sp = "dev"
        else:
            sp = "train"
        for spk in show_spks[sh]:
            split_of[spk] = sp
        counts[sp] += len(show_spks[sh])

    split_shows = defaultdict(set)
    for spk, sp in split_of.items():
        split_shows[sp].add(spk_show[spk])
    for a_, b_ in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        inter = split_shows[a_] & split_shows[b_]
        assert not inter, f"Fuite : émissions communes {a_}/{b_} : {inter}"
    for sp in ("train", "dev", "eval"):
        if counts[sp] < 2:
            logging.warning(f"Split {sp} : {counts[sp]} locuteur(s) -> S02 impossible intra-split.")
    logging.info(
        f"Splits locuteurs {counts} | émissions { {k: len(v) for k, v in split_shows.items()} }"
    )

    # folds groupés par émission (donc aussi par locuteur)
    k = min(5, len(shows))
    folds = []
    for fold_i in range(k):
        eval_shows = set(shows[fold_i::k])
        eval_spks = sorted(s_ for s_ in spk_show if spk_show[s_] in eval_shows)
        train_spks = sorted(s_ for s_ in spk_show if s_ not in eval_spks)
        folds.append(
            {"fold": fold_i, "train": train_spks, "eval": eval_spks, "eval_shows": sorted(eval_shows)}
        )

    refs = {r["spk_id"]: r for r in records if r["is_ref"]}
    same_split = defaultdict(list)
    for spk, spl in split_of.items():
        same_split[spl].append(spk)
    rot = defaultdict(int)

    for rec in records:
        rec["split"] = split_of[rec["spk_id"]]
        rec["ref_s01"] = refs[rec["spk_id"]]["utt_id"]
        others = sorted(s_ for s_ in same_split[rec["split"]] if s_ != rec["spk_id"])
        if others:
            # rotation sur les voix cibles du même split (diversité S02)
            other = others[rot[rec["spk_id"]] % len(others)]
            if not rec["is_ref"]:
                rot[rec["spk_id"]] += 1
            rec["ref_s02_spk"] = other
            rec["ref_s02"] = refs[other]["utt_id"]
        else:
            rec["ref_s02_spk"] = rec["spk_id"]
            rec["ref_s02"] = rec["ref_s01"]

    return records, split_of, folds


def to_mono_16k(wav, sr):
    if isinstance(wav, np.ndarray):
        wav = torch.tensor(wav, dtype=torch.float32)
    else:
        wav = wav.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != Config.TARGET_SR:
        wav = AF.resample(wav, sr, Config.TARGET_SR)
    return wav.squeeze(0)


def energy_crop(wav, target_len=Config.TARGET_LEN):
    """Crop 5 s sur la fenêtre d'énergie max. Pas de zero-padding."""
    n = wav.numel()
    if n <= target_len:
        return wav
    hop = max(1, Config.TARGET_SR // 100)
    windows = wav.unfold(0, target_len, hop)
    energy = windows.pow(2).mean(dim=1)
    idx = int(torch.argmax(energy).item()) * hop
    idx = min(idx, n - target_len)
    return wav[idx : idx + target_len]


def peak_norm(wav, peak_target=0.708):
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * peak_target
    return wav


def channel_degrade(wav, seed_key):
    """Même chaîne pour bona fide et spoof : passe-bande + bruit blanc dont le SNR
    est tiré de la même loi (seed déterministe par utt_id). Réduit le raccourci canal."""
    if not Config.CHANNEL_MATCH:
        return wav
    x = wav.unsqueeze(0)
    x = AF.highpass_biquad(x, Config.TARGET_SR, Config.CHANNEL_HP_HZ)
    x = AF.lowpass_biquad(x, Config.TARGET_SR, Config.CHANNEL_LP_HZ)
    rng = np.random.RandomState(zlib.crc32(seed_key.encode("utf-8")))
    snr_db = rng.uniform(*Config.CHANNEL_SNR_DB)
    sig_pow = x.pow(2).mean().clamp_min(1e-12)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = torch.from_numpy(rng.randn(x.shape[-1]).astype(np.float32)).unsqueeze(0)
    x = x + noise * noise_pow.sqrt()
    return x.squeeze(0)


def postprocess_wav(audio_array, sr_in, seed_key="x"):
    wav = to_mono_16k(audio_array, sr_in)
    wav = energy_crop(wav, Config.TARGET_LEN)
    if wav.numel() < Config.MIN_SAMPLES:
        return None
    wav = channel_degrade(wav, seed_key)
    wav = peak_norm(wav)
    return wav


def postprocess_and_save(audio_array, sr_in, out_path):
    # seed sur l'utt_id sans préfixe S0x_ : même tirage pour un bona fide et ses spoofs
    stem = re.sub(r"^S0\d_", "", os.path.splitext(os.path.basename(out_path))[0])
    wav = postprocess_wav(audio_array, sr_in, seed_key=stem)
    if wav is None:
        logging.warning(f"Audio trop court après DSP, ignoré : {out_path}")
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, wav.detach().cpu().numpy(), Config.TARGET_SR, subtype="PCM_16")
    return wav


def load_slice(src_wav, start, end):
    info = sf.info(src_wav)
    sr = info.samplerate
    i0 = max(0, int(round(start * sr)))
    i1 = min(info.frames, int(round(end * sr)))
    if i1 <= i0:
        return None, sr
    wav, _ = sf.read(src_wav, start=i0, stop=i1, dtype="float32", always_2d=False)
    return wav, sr


def export_bonafide(records):
    exported = []
    for rec in tqdm(records, desc="Export bona fide / refs"):
        wav, sr = load_slice(rec["src_wav"], rec["start"], rec["end"])
        if wav is None:
            continue
        if rec["is_ref"]:
            out_path = os.path.join(Config.REF_DIR, f"{rec['utt_id']}.wav")
        else:
            out_path = os.path.join(Config.BONAFIDE, f"{rec['utt_id']}.wav")
        saved = postprocess_and_save(wav, sr, out_path)
        if saved is None:
            continue
        rec = dict(rec)
        rec["wav_path"] = out_path.replace("\\", "/")
        rec["n_samples"] = int(saved.numel())
        rec["sr"] = Config.TARGET_SR
        exported.append(rec)
    n_b = sum(1 for r in exported if not r["is_ref"])
    n_r = sum(1 for r in exported if r["is_ref"])
    logging.info(f"Export : {n_b} bona fide -> {Config.BONAFIDE} | {n_r} refs -> {Config.REF_DIR}")
    return exported


def save_manifest(records, speaker_map, split_of, folds, clips_per):
    payload = {
        "source": Config.SPEECH_DATASET_DIR,
        "clips_per_speaker": clips_per,
        "speaker_map": speaker_map,
        "split_of": split_of,
        "records": records,
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH) or ".", exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.makedirs(Config.PROTO_DIR, exist_ok=True)
    with open(FOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"k": len(folds), "folds": folds}, f, ensure_ascii=False, indent=2)
    logging.info(f"Manifeste : {MANIFEST_PATH}")
    logging.info(f"Folds locuteur/émission : {FOLDS_PATH}")
    return payload


def load_manifest():
    if not os.path.isfile(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def content_records(manifest):
    return [r for r in manifest["records"] if not r.get("is_ref")]


def ref_path(utt_id):
    return os.path.join(Config.REF_DIR, f"{utt_id}.wav")


def download_xtts_checkpoint():
    if (
        os.path.isfile(Config.XTTS_CHECKPOINT)
        and os.path.isfile(Config.XTTS_CONFIG)
        and os.path.isfile(Config.XTTS_VOCAB)
    ):
        logging.info(f"Checkpoint Anta déjà présent : {Config.XTTS_CHECKPOINT_DIR}")
        return

    if not Config.HF_TOKEN:
        raise RuntimeError(
            "Checkpoint XTTS manquant et HF_TOKEN / HUGGING_FACE_HUB_TOKEN non défini. "
            f"Attendu : {Config.XTTS_CHECKPOINT}"
        )

    import yaml

    login(token=Config.HF_TOKEN, add_to_git_credential=False)
    yml_path = hf_hub_download(
        repo_id=Config.XTTS_REPO,
        filename="checkpoint-id.yml",
        local_dir=Config.MODEL_DIR,
        token=Config.HF_TOKEN,
    )
    with open(yml_path, "r", encoding="utf-8") as f:
        ckpt_info = yaml.safe_load(f)

    if isinstance(ckpt_info, dict):
        gdrive_id = ckpt_info.get("id") or ckpt_info.get("gdrive_id") or list(ckpt_info.values())[0]
    else:
        gdrive_id = str(ckpt_info).strip()

    zip_path = os.path.join(Config.MODEL_DIR, "galsenai-xtts-wo-checkpoints.zip")
    if not os.path.exists(zip_path):
        logging.info("Téléchargement du checkpoint XTTS (~7 Go) via gdown...")
        subprocess.run([sys.executable, "-m", "gdown", gdrive_id, "-O", zip_path], check=True)

    extract_marker = os.path.join(Config.MODEL_DIR, ".extracted")
    if not os.path.exists(extract_marker):
        logging.info("Extraction du zip XTTS...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(Config.MODEL_DIR)
        with open(extract_marker, "w") as f:
            f.write("ok")
        logging.info("Extraction terminée.")

    for path in [Config.XTTS_CHECKPOINT, Config.XTTS_CONFIG, Config.XTTS_VOCAB]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Fichier XTTS toujours introuvable après download : {path}")


def _safe_torchaudio_load(filepath, *args, **kwargs):
    if isinstance(filepath, list):
        filepath = filepath[0]
    wav, sr = sf.read(filepath, dtype="float32")
    wav_tensor = torch.from_numpy(wav)
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    else:
        wav_tensor = wav_tensor.t()
    return wav_tensor, sr


torchaudio.load = _safe_torchaudio_load


def load_xtts_model():
    local_tts_root = os.path.abspath(Config.WOLOF_TTS_ROOT)
    if not os.path.isdir(os.path.join(local_tts_root, "TTS")):
        raise FileNotFoundError(
            f"Code TTS local introuvable : {os.path.join(local_tts_root, 'TTS')} "
            "(définir WOLOF_TTS_ROOT)"
        )
    if local_tts_root in sys.path:
        sys.path.remove(local_tts_root)
    sys.path.insert(0, local_tts_root)

    import TTS

    print(f"[XTTS] Bibliothèque TTS utilisée : {TTS.__file__}")
    loaded_tts_path = os.path.normcase(os.path.abspath(TTS.__file__))
    expected_tts_path = os.path.normcase(os.path.join(local_tts_root, "TTS", "__init__.py"))
    if loaded_tts_path != expected_tts_path:
        raise RuntimeError(
            "Python utilise une autre bibliothèque TTS. "
            f"Chemin trouvé : {TTS.__file__}"
        )

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    for path in [Config.XTTS_CHECKPOINT, Config.XTTS_CONFIG, Config.XTTS_VOCAB]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Fichier XTTS introuvable : {path}")

    logging.info(f"Chargement Anta depuis : {Config.XTTS_CHECKPOINT_DIR}")
    config = XttsConfig()
    config.load_json(Config.XTTS_CONFIG)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=Config.XTTS_CHECKPOINT,
        vocab_path=Config.XTTS_VOCAB,
        use_deepspeed=False,
    )
    model.to(Config.DEVICE)
    model.eval()
    print(
        f"[XTTS] Limite de texte wolof : "
        f"{model.tokenizer.char_limits.get('wo', 250)} caractères"
    )
    print("[XTTS] Modèle Anta_GPT_XTTS_Wo chargé.")
    return model, config


def synth_xtts(model, text, speaker_wav, gpt_cond_latent=None, speaker_embedding=None):
    with torch.inference_mode():
        if gpt_cond_latent is None or speaker_embedding is None:
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[speaker_wav],
                gpt_cond_len=model.config.gpt_cond_len,
                max_ref_length=model.config.max_ref_len,
                sound_norm_refs=model.config.sound_norm_refs,
            )
        result = model.inference(
            text=text.lower(),
            language="wo",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            do_sample=False,
            speed=1.06,
            enable_text_splitting=True,
        )
        wav = torch.tensor(result["wav"]).unsqueeze(0)
    return wav, Config.XTTS_SR


def _cache_xtts_latents(model, cache, wav_path):
    if wav_path not in cache:
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Référence XTTS manquante : {wav_path}")
        with torch.inference_mode():
            cache[wav_path] = model.get_conditioning_latents(
                audio_path=[wav_path],
                gpt_cond_len=model.config.gpt_cond_len,
                max_ref_length=model.config.max_ref_len,
                sound_norm_refs=model.config.sound_norm_refs,
            )
    return cache[wav_path]


def generate_s01_s02(model, records):
    items = [r for r in records if not r.get("is_ref")]
    if not items:
        logging.error("Aucun énoncé contenu pour S01/S02.")
        return
    latent_cache = {}
    logging.info("Génération S01 (clone même locuteur, ref hors énoncé) ...")
    for rec in tqdm(items, desc="S01"):
        out_path = os.path.join(Config.S01_DIR, f"S01_{rec['utt_id']}.wav")
        if os.path.exists(out_path):
            continue
        try:
            ref = ref_path(rec["ref_s01"])
            lat, emb = _cache_xtts_latents(model, latent_cache, ref)
            wave_arr, sr_out = synth_xtts(model, rec["text_tts"], ref, lat, emb)
            postprocess_and_save(wave_arr, sr_out, out_path)
        except Exception as e:
            logging.error(f"Échec S01 sur {rec['utt_id']} : {e}")

    logging.info("Génération S02 (autre locuteur, même split, rotation) ...")
    for rec in tqdm(items, desc="S02"):
        out_path = os.path.join(Config.S02_DIR, f"S02_{rec['utt_id']}.wav")
        if os.path.exists(out_path):
            continue
        try:
            ref = ref_path(rec["ref_s02"])
            lat, emb = _cache_xtts_latents(model, latent_cache, ref)
            wave_arr, sr_out = synth_xtts(model, rec["text_tts"], ref, lat, emb)
            postprocess_and_save(wave_arr, sr_out, out_path)
        except Exception as e:
            logging.error(f"Échec S02 sur {rec['utt_id']} : {e}")


def _load_speecht5_speaker_embeddings(device):
    """Plusieurs x-vectors CMU-Arctic (un par locuteur). Pas de fallback aléatoire."""
    from datasets import load_dataset

    logging.info("Chargement des speaker embeddings SpeechT5 (CMU-Arctic xvectors)...")
    xvectors = load_dataset(Config.SPEECHT5_XVECTORS, split="validation", trust_remote_code=True)
    return [
        torch.tensor(xvectors[i]["xvector"]).unsqueeze(0).to(device)
        for i in Config.SPEECHT5_XVECTOR_IDX
    ]


def generate_s03(records):
    items = [r for r in records if not r.get("is_ref")]
    if not items:
        logging.error("Aucun énoncé contenu pour S03.")
        return

    logging.info(f"Chargement SpeechT5 Wolof ({Config.SPEECHT5_REPO})...")
    processor = SpeechT5Processor.from_pretrained(Config.SPEECHT5_REPO)
    model_st5 = SpeechT5ForTextToSpeech.from_pretrained(Config.SPEECHT5_REPO).to(Config.DEVICE)
    model_st5.eval()
    logging.info(f"Chargement vocoder HiFi-GAN ({Config.SPEECHT5_VOCODER})...")
    vocoder = SpeechT5HifiGan.from_pretrained(Config.SPEECHT5_VOCODER).to(Config.DEVICE)
    vocoder.eval()
    speaker_embs = _load_speecht5_speaker_embeddings(Config.DEVICE)
    st5_sr = 16000

    logging.info("Génération S03 (SpeechT5 Wolof, un x-vector par locuteur) ...")
    for rec in tqdm(items, desc="S03 SpeechT5"):
        out_path = os.path.join(Config.S03_DIR, f"S03_{rec['utt_id']}.wav")
        if os.path.exists(out_path):
            continue
        try:
            inputs = processor(text=rec["text_tts"], return_tensors="pt")
            input_ids = inputs["input_ids"].to(Config.DEVICE)
            emb = speaker_embs[int(re.sub(r"\D", "", rec["spk_id"])) % len(speaker_embs)]
            with torch.inference_mode():
                speech = model_st5.generate_speech(input_ids, emb, vocoder=vocoder)
            postprocess_and_save(speech.cpu().numpy(), st5_sr, out_path)
        except Exception as e:
            logging.error(f"Échec S03 sur {rec['utt_id']} : {e}")

    del model_st5, vocoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_protocol(manifest):
    items = content_records(manifest)
    split_lines = {"train": [], "dev": [], "eval": []}
    attacks = (("S01", Config.S01_DIR), ("S02", Config.S02_DIR), ("S03", Config.S03_DIR))
    active = [t for t, d in attacks if glob.glob(os.path.join(d, "*.wav"))]
    failures = []
    for rec in items:
        spk, utt, spl = rec["spk_id"], rec["utt_id"], rec["split"]
        wavs = {
            tag: os.path.join(folder, f"{tag}_{utt}.wav")
            for tag, folder in attacks
            if tag in active
        }
        missing = [t for t, w in wavs.items() if not os.path.isfile(w)]
        bona_ok = os.path.isfile(os.path.join(Config.BONAFIDE, f"{utt}.wav"))
        if missing or not bona_ok:
            failures.append(
                {"utt_id": utt, "spk_id": spk, "missing": missing, "bonafide_ok": bona_ok}
            )
            continue  # trio incomplet : exclu du protocole (appariement strict)
        split_lines[spl].append(f"{spk}  {utt}  -  -  bonafide")
        for tag in active:
            split_lines[spl].append(f"{spk}  {tag}_{utt}  {tag}  -  spoof")

    fail_path = os.path.join(os.path.dirname(MANIFEST_PATH), "spoof_failures.json")
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(
            {"n_excluded": len(failures), "n_total": len(items), "items": failures},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info(f"{len(failures)}/{len(items)} énoncés exclus (trio incomplet) -> {fail_path}")

    written = {}
    for split, lines in split_lines.items():
        proto_out = os.path.join(Config.PROTO_DIR, f"wolof_deepfake.{split}.txt")
        with open(proto_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        written[split] = lines
        logging.info(f"Protocole {split}: {len(lines)} lignes -> {proto_out}")

    all_lines = []
    for split in ("train", "dev", "eval"):
        all_lines.extend(split_lines[split])
    all_path = os.path.join(Config.PROTO_DIR, "wolof_deepfake.all.txt")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
        if all_lines:
            f.write("\n")
    return written


def write_metadata(manifest, splits):
    n_b = len(glob.glob(os.path.join(Config.BONAFIDE, "*.wav")))
    n1 = len(glob.glob(os.path.join(Config.S01_DIR, "*.wav")))
    n2 = len(glob.glob(os.path.join(Config.S02_DIR, "*.wav")))
    n3 = len(glob.glob(os.path.join(Config.S03_DIR, "*.wav")))
    summary = {
        "dataset_name": "Wolof Audio Deepfake Detection Dataset (STM)",
        "source": Config.SPEECH_DATASET_DIR,
        "device": Config.DEVICE,
        "bonafide_dir": Config.BONAFIDE,
        "spoof_dir": Config.SPOOF_DIR,
        "total_fichiers": n_b + n1 + n2 + n3,
        "bonafide": n_b,
        "spoof": {
            "S01_xTTS_clone": n1,
            "S02_xTTS_multi_ref": n2,
            "S03_SpeechT5": n3,
        },
        "speakers": manifest.get("speaker_map", {}),
        "split_of": manifest.get("split_of", {}),
        "protocol_counts": {k: len(v) for k, v in (splits or {}).items()},
        "notes": {
            "old_bonafide_unused_for_train": Config.BONAFIDE_OLD,
            "no_zero_padding": True,
            "channel_match": Config.CHANNEL_MATCH,
            "report_eer_per_attack": True,
            "s02_refs_intra_split": True,
            "splits_disjoint_by_show": True,
            "s02_target_rotation": True,
            "s03_xvectors_per_speaker": Config.SPEECHT5_XVECTOR_IDX,
            "french_codeswitch_filtered": True,
            "external_test_70_clips": Config.BONAFIDE_OLD,
            "external_spoof_set": f"{Config.BASE_DIR}/external_test/spoof_wild (à constituer)",
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Métadonnées : {METADATA_PATH}")
    return summary


def _wav_features(path, head_s=0.3):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n_head = min(len(wav), int(head_s * sr))
    head = wav[: max(1, n_head)]
    rms = float(np.sqrt(np.mean(head**2) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.sign(head)))) / 2.0)
    spec = np.abs(np.fft.rfft(head * np.hanning(len(head))))
    freqs = np.fft.rfftfreq(len(head), 1.0 / sr)
    mag = spec + 1e-12
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    energy = wav**2
    thr = np.percentile(energy, 20)
    noise = float(np.mean(energy[energy <= thr]) + 1e-12)
    sig = float(np.mean(energy[energy >= np.percentile(energy, 80)]) + 1e-12)
    snr = float(10 * math.log10(sig / noise))
    cum = np.cumsum(mag)
    band = float(freqs[int(np.searchsorted(cum, 0.85 * cum[-1]))]) if len(freqs) else 0.0
    return {
        "rms_0_3s": rms,
        "zcr_0_3s": zcr,
        "centroid_0_3s": centroid,
        "snr_proxy": snr,
        "bandwidth_85": band,
    }


def run_channel_qc(max_per_class=200):
    bona = sorted(glob.glob(os.path.join(Config.BONAFIDE, "*.wav")))
    spoof = []
    for folder in (Config.S01_DIR, Config.S02_DIR, Config.S03_DIR):
        spoof.extend(sorted(glob.glob(os.path.join(folder, "*.wav"))))
    rng = random.Random(42)
    if len(bona) > max_per_class:
        bona = rng.sample(bona, max_per_class)
    if len(spoof) > max_per_class:
        spoof = rng.sample(spoof, max_per_class)

    def mean_feats(paths):
        rows = []
        for p in paths:
            try:
                rows.append(_wav_features(p))
            except Exception:
                continue
        if not rows:
            return {}, []
        keys = rows[0].keys()
        avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        return avg, rows

    avg_b, rows_b = mean_feats(bona)
    avg_s, rows_s = mean_feats(spoof)

    keys = ["rms_0_3s", "zcr_0_3s", "centroid_0_3s", "snr_proxy", "bandwidth_85"]

    def probe(rb, rs):
        """Sonde linéaire (moindres carrés) validée en 2-fold ; acc ~0.5 = pas de raccourci."""
        if len(rb) < 6 or len(rs) < 6:
            return None
        x = np.array([[r[k] for k in keys] for r in rb + rs])
        y = np.concatenate([np.zeros(len(rb)), np.ones(len(rs))])
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        x = np.hstack([x, np.ones((len(x), 1))])
        idx = np.random.RandomState(0).permutation(len(x))
        halves = [idx[::2], idx[1::2]]
        accs = []
        for a, b in ((0, 1), (1, 0)):
            w, _, _, _ = np.linalg.lstsq(x[halves[a]], y[halves[a]], rcond=None)
            accs.append(float(np.mean(((x[halves[b]] @ w) >= 0.5) == y[halves[b]])))
        return float(np.mean(accs))

    acc = probe(rows_b, rows_s)
    acc_by_attack = {}
    for tag, folder in (("S01", Config.S01_DIR), ("S02", Config.S02_DIR), ("S03", Config.S03_DIR)):
        paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
        if len(paths) > max_per_class:
            paths = rng.sample(paths, max_per_class)
        _, rows_a = mean_feats(paths)
        acc_by_attack[tag] = probe(rows_b, rows_a)

    report = {
        "n_bonafide_sampled": len(rows_b),
        "n_spoof_sampled": len(rows_s),
        "mean_bonafide": avg_b,
        "mean_spoof": avg_s,
        "linear_probe_acc": acc,
        "linear_probe_acc_by_attack": acc_by_attack,
        "channel_match": Config.CHANNEL_MATCH,
        "warning": None,
    }
    worst = max([a for a in acc_by_attack.values() if a is not None] + ([acc] if acc else [0]))
    if worst >= 0.8:
        report["warning"] = (
            "Une sonde linéaire (features canal) sépare encore bien bona/spoof "
            f"(acc max={worst:.2f}, par attaque={acc_by_attack}) : risque de raccourci canal. "
            "Documenter ou resserrer codec/bande."
        )
        logging.warning(report["warning"])
    elif acc is None:
        logging.info("QC canal : pas assez de spoofs pour la sonde.")
    else:
        logging.info(f"QC canal : acc={acc:.2f} par attaque={acc_by_attack} (alerte >= 0,80).")

    with open(QC_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"Rapport QC : {QC_PATH}")
    return report


def yield_report(segments):
    """Rendement des segments éligibles avant de figer N_SPEAKERS x CLIPS_PER_SPEAKER."""
    by_spk = defaultdict(list)
    for seg in segments:
        by_spk[seg["raw_spk"]].append(seg)
    need = Config.CLIPS_PER_SPEAKER + Config.CLIP_MARGIN + 1
    rows = []
    for spk, segs in by_spk.items():
        rows.append(
            {
                "raw_spk": spk,
                "n_segments": len(segs),
                "n_checked": sum(1 for s in segs if s["checked"]),
                "shows": sorted({s["show_id"] for s in segs}),
                "gender": next((s["gender"] for s in segs if s["gender"] != "unknown"), "unknown"),
                "total_dur_s": round(sum(s["duration"] for s in segs), 1),
                "mean_dur_s": round(float(np.mean([s["duration"] for s in segs])), 2),
                "eligible": len(segs) >= need,
            }
        )
    rows.sort(key=lambda r: -r["n_segments"])
    eligible = [r for r in rows if r["eligible"]]
    durs = [s["duration"] for s in segments]
    report = {
        "min_seg_dur": Config.MIN_SEG_DUR,
        "max_seg_dur": Config.MAX_SEG_DUR,
        "needed_per_speaker": need,
        "n_segments": len(segments),
        "n_speakers_total": len(rows),
        "n_speakers_eligible": len(eligible),
        "eligible_by_gender": {
            g: sum(1 for r in eligible if r["gender"] == g) for g in ("female", "male", "unknown")
        },
        "eligible_shows": sorted({sh for r in eligible for sh in r["shows"]}),
        "duration_hist_s": {
            "mean": round(float(np.mean(durs)), 2) if durs else None,
            "p10": round(float(np.percentile(durs, 10)), 2) if durs else None,
            "p50": round(float(np.percentile(durs, 50)), 2) if durs else None,
            "p90": round(float(np.percentile(durs, 90)), 2) if durs else None,
        },
        "feasible": len(eligible) >= Config.N_SPEAKERS,
        "speakers": rows[:40],
    }
    path = os.path.join(Config.BASE_DIR, "yield_report.json")
    os.makedirs(Config.BASE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(
        f"Rendement : {len(segments)} segments, {len(rows)} locuteurs, "
        f"{len(eligible)} éligibles (>= {need} segs) -> faisable={report['feasible']} | {path}"
    )
    return report


def apply_pilot():
    """Mini-run (8 locuteurs, 5+1 clips) dans data/pilot/ pour écoute et QC."""
    Config.N_SHOWS = 8
    Config.N_SPEAKERS = 8
    Config.CLIPS_PER_SPEAKER = 5
    Config.CLIP_MARGIN = 1
    Config.MAX_SPK_PER_SHOW = 2
    root = f"{Config.BASE_DIR}/pilot"
    Config.BONAFIDE = f"{root}/bonafide_new"
    Config.REF_DIR = f"{root}/refs_new"
    Config.SPOOF_DIR = f"{root}/spoof_new"
    Config.PROTO_DIR = f"{root}/protocols_new"
    Config.S01_DIR = f"{Config.SPOOF_DIR}/S01_xtts_wolof"
    Config.S02_DIR = f"{Config.SPOOF_DIR}/S02_xtts_multi_ref"
    Config.S03_DIR = f"{Config.SPOOF_DIR}/S03_speecht5_wolof"
    global MANIFEST_PATH, METADATA_PATH, QC_PATH, FOLDS_PATH
    MANIFEST_PATH = f"{root}/manifest_stm.json"
    METADATA_PATH = f"{root}/dataset_metadata_new.json"
    QC_PATH = f"{root}/qc_channel_shortcut.json"
    FOLDS_PATH = f"{Config.PROTO_DIR}/speaker_folds.json"
    logging.info(f"MODE PILOTE -> {root}")


def prepare_corpus():
    setup_directories()
    segments = parse_all_segments()
    records, speaker_map, clips_per = select_minimal_utterances(segments)
    records, split_of, folds = assign_splits_and_refs(records, speaker_map)
    exported = export_bonafide(records)
    keep_ids = {r["utt_id"] for r in exported}
    exported_full = [r for r in records if r["utt_id"] in keep_ids]
    missing = [r["utt_id"] for r in records if r["utt_id"] not in keep_ids]
    if missing:
        logging.warning(f"{len(missing)} extraits non exportés (audio trop court / lecture).")
    return save_manifest(exported_full, speaker_map, split_of, folds, clips_per)


def parse_args():
    p = argparse.ArgumentParser(description="Base wolof bona fide / spoof depuis speech_dataset")
    p.add_argument("--prepare-only", action="store_true", help="STM + découpe + protocole, sans TTS")
    p.add_argument("--skip-xtts", action="store_true")
    p.add_argument("--skip-s03", action="store_true")
    p.add_argument("--qc-only", action="store_true")
    p.add_argument("--yield-report", action="store_true", help="Compte les segments éligibles et quitte")
    p.add_argument("--pilot", action="store_true", help="Mini-run 8 locuteurs / 5 clips")
    p.add_argument("--no-channel", action="store_true", help="Désactive la dégradation canal commune")
    p.add_argument("--threads", type=int, default=Config.CPU_THREADS, help="Nombre de threads CPU PyTorch")
    p.add_argument("--limit", type=int, default=0, help="Limite le nombre d'énoncés à synthétiser (test)")
    return p.parse_args()


def main():
    args = parse_args()
    if Config.DEVICE == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
    if not os.environ.get("HF_ONLINE"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if args.no_channel:
        Config.CHANNEL_MATCH = False
    if args.pilot:
        apply_pilot()
    if args.yield_report:
        yield_report(parse_all_segments())
        return
    logging.info(f"Device : {Config.DEVICE} (threads CPU : {torch.get_num_threads()})")

    if args.qc_only:
        if not load_manifest():
            logging.error("Pas de manifeste. Lancer d'abord sans --qc-only.")
            return
        splits = build_protocol(load_manifest())
        write_metadata(load_manifest(), splits)
        run_channel_qc()
        return

    manifest = load_manifest()
    if manifest is None or args.prepare_only:
        manifest = prepare_corpus()
    else:
        logging.info(f"Manifeste existant réutilisé : {MANIFEST_PATH}")
        setup_directories()

    if not args.prepare_only:
        records = manifest["records"]
        if args.limit > 0:
            non_refs = [r for r in records if not r.get("is_ref")][: args.limit]
            refs = [r for r in records if r.get("is_ref")]
            records = non_refs + refs
            logging.info(f"Mode --limit {args.limit} : {len(non_refs)} énoncés sélectionnés.")
        if not args.skip_s03:
            try:
                generate_s03(records)
            except Exception:
                logging.error("Erreur SpeechT5", exc_info=True)
        if not args.skip_xtts:
            try:
                download_xtts_checkpoint()
                logging.info("Chargement Anta XTTS...")
                model_xtts, _ = load_xtts_model()
                generate_s01_s02(model_xtts, records)
                del model_xtts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logging.error("Erreur XTTS", exc_info=True)

    splits = build_protocol(manifest)
    write_metadata(manifest, splits)
    run_channel_qc()
    n_b = len(glob.glob(os.path.join(Config.BONAFIDE, "*.wav")))
    n_s = (
        len(glob.glob(os.path.join(Config.S01_DIR, "*.wav")))
        + len(glob.glob(os.path.join(Config.S02_DIR, "*.wav")))
        + len(glob.glob(os.path.join(Config.S03_DIR, "*.wav")))
    )
    logging.info(f"Terminé. Bona fide={n_b} | spoof={n_s}")


if __name__ == "__main__":
    main()
