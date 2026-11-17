# Effet de la fuite de voix (speech_dataset, 10 « locuteurs » actuels)

Scripts : `audit/leak_groups.py`, `audit/leak_effect.py`, `cv_pretrained_group.py`, `cv_pretrained_group2.py` ; sorties : `audit/leak_effect.json`, `experiments_pretrained/equalized/cv_speakers_grp{,2}/results.json`.

## Constat
- Les « locuteurs » du manifeste = émission + identifiant local (ex. `8_locuteur_3`) : plusieurs personnes par « locuteur » (similarité intra 0,10–0,60 pour 7 sur 10) et genres mélangés.
- 3 paires partagent une voix entre splits : SPK06 (eval)–SPK09 (train), SPK02 (dev)–SPK07 (train), SPK00 (train)–SPK05 (dev).
- Part des clips de test dont la voix est aussi dans l'entraînement : 22–64 % (LCNN, seuil 0,5) ; 40–85 % (sondes).

## Mesure : sondes réentraînées en retirant les clips d'entraînement dont la voix est dans le test (EER moyen par pli, 5 plis)
| Modèle | Actuel | Témoin : même nb de clips retirés au hasard | Sans voix partagées | Écart dû aux voix |
|---|---|---|---|---|
| WavLM, seuil 0,5 | 1,07 | 2,66 ± 1,88 | 5,25 ± 2,70 | +2,6 |
| WavLM, seuil 0,8 | 1,07 | 2,04 ± 2,63 | 3,86 ± 2,50 | +1,8 |
| wav2vec2, seuil 0,5 | 0,85 | 3,58 ± 1,87 | 5,14 ± 2,81 | +1,6 |
| wav2vec2, seuil 0,8 | 0,85 | 2,63 ± 2,51 | 5,70 ± 4,51 | +3,1 |
Clips d'entraînement : ~810 → 124–279 (0,5) ou 182–398 (0,8).

## Lecture
- La perte de données seule fait passer l'EER de ~1 % à ~2–3,6 % (témoin).
- Retirer les voix partagées ajoute environ +1,6 à +3 points : plus de pli en défaut qu'en gain (WavLM 0,5 : 4/5 ; wav2vec2 0,8 : 5/5), mais écarts-types 2–4,5 sur 5 plis : **non significatif**.
- Les chiffres « ~1 % » sont optimistes ; un ordre de grandeur honnête pour des voix inconnues et ~200–400 clips d'entraînement est **3–6 %**.
- Limites : retirer une voix retire aussi son émission/canal (voix et canal non séparables) ; regroupement sans vérité terrain ; LCNN non réentraîné ; pas de graines multiples.
