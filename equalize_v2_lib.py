"""Etape finale d'egalisation v2 (apres equalize_channel.process_one) : retire le decalage continu (DC) residuel du limiteur doux et aligne la
dynamique de volume (P95-P10 de l'RMS en dB, meme mesure que la sonde G1 'dyn_db') par ajout de bruit de canal mis en forme, avec une cible commune
(17,25 dB +/- 0,8, tirage deterministe par fichier) pour TOUTES les classes. ATTENTION : cette mesure est aussi une sonde ; l'aligner ne prouve pas
l'absence d'autres raccourcis."""
import numpy as np, librosa
from scipy import signal
import equalize_channel as EQ

GOAL, GOAL_SD = 17.25, 0.8


def dyn_db(x):
    fr = librosa.feature.rms(y=x.astype(np.float32), frame_length=400, hop_length=160)[0]
    f = 20 * np.log10(fr + 1e-10)
    return float(np.percentile(f, 95) - np.percentile(f, 10))


def finalize(x, target_mag, sos, seed):
    rng = np.random.RandomState(seed)
    goal = GOAL + rng.normal(0.0, GOAL_SD)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    if dyn_db(x) > goal:
        w = signal.sosfiltfilt(sos, rng.randn(len(x) + 400))[: len(x)]
        w = EQ.apply_ltas_match(w, target_mag, rng, jitter_std=0.01)
        w = w / (np.sqrt(np.mean(w ** 2)) + 1e-9) * rms
        lo, hi = 0.0, 1.5
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if dyn_db(x + mid * w) > goal:
                lo = mid
            else:
                hi = mid
        x = x + 0.5 * (lo + hi) * w
    x = x - np.mean(x)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10.0 ** ((-20.0 + rng.normal(0.0, 0.5)) / 20.0)
    x = np.clip(x, -0.98, 0.98)
    return x - np.mean(x)
