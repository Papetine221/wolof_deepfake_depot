# Détection de deepfakes audio en wolof

Code, protocoles et résultats du mémoire de Master 2 Informatique (UGB, UFR SAT) « Détection de deepfakes audio en wolof : constitution d'une base de données et évaluation comparative de détecteurs ».

**Aucun fichier audio n'est fourni.** La base contient des voix de personnes réelles (corpus Kallaama) et des faux qui en sont dérivés : elle n'est pas redistribuée. Les scripts permettent de la reconstruire à partir du corpus d'origine et des modèles publics.

## La base

- 71 personnes, 616 énoncés authentiques, 2 266 faux (2 882 fichiers).
- Voix authentiques : corpus Kallaama (CC BY 4.0), regroupées par personne avec des empreintes ECAPA-TDNN.
- Quatre systèmes de faux :
  - S01 : XTTS v2 adapté au wolof, voix de la personne de l'énoncé ;
  - S02 : XTTS v2 adapté au wolof, voix d'une autre personne du même pli ;
  - S03 : SpeechT5 pour le wolof ;
  - S05 : conversion de voix kNN-VC.
- Tous les fichiers passent par la même chaîne de dégradation et d'égalisation.
- Évaluation en validation croisée à 5 plis de personnes.

Dans les noms de fichiers, « v2 » désigne cette base finale. Les fichiers sans « v2 » (`data/manifest_stm.json`, `data/protocols_new/`, `experiments_pretrained/equalized/`) concernent la base préliminaire, utilisée seulement pour le contrôle de la fuite de voix.

## Contenu du dépôt

| Étape | Fichiers |
|---|---|
| Empreintes de locuteur et regroupement en personnes | `audit/spk_embed_speech_dataset.py`, `audit/cluster_speakers.py`, `audit/build_group_folds.py` |
| Protocoles | `protocols_groups/persons_t050.csv` (couple fichier/locuteur → personne), `protocols_groups/folds_t050.json` (5 plis de personnes) |
| Manifeste de la base | `build_v2_manifest.py` → `data_v2/manifest_v2.json` |
| Génération des faux (Colab) | `generation_v2_full.py` (utilise `generation_faux.py`) |
| Égalisation | `equalize_v2_full.py`, `equalize_v2_lib.py`, `equalize_channel.py` |
| Contrôle des raccourcis | `audit/v2_probes.py` |
| Extraction des représentations | `extract_pretrained_v2.py` (wav2vec 2.0, WavLM, XLS-R, MMS) ; `dl_xl.py` télécharge XLS-R et MMS |
| Sondes gelées, plis, générateur laissé de côté | `cv_v2_probe.py`, `cv_v2_logo_full.py` |
| Courbe d'apprentissage | `cv_v2_curve.py` |
| LCNN-Mel | `cv_v2_lcnn.py` (hyperparamètres de `train_all.py`, modèle dans `src/`) |
| Intervalles de confiance, seuil fixe | `audit/metrics_extra.py` |
| Test exploratoire Oolel (S04) | `audit/test_s04_oolel.py` (WavLM), `audit/test_s04_oolel_xlsr.py` (XLS-R) |
| Contrôle de la fuite de voix (base préliminaire) | `extract_pretrained.py`, `cv_pretrained_group.py`, `cv_pretrained_group2.py` (WavLM, wav2vec 2.0) ; `extract_pretrained_xl_prelim.py`, `cv_pretrained_group_xl.py` (XLS-R, MMS) ; `audit/leak_groups.py` |
| Figures du mémoire | `audit/make_figures_v2.py` |

Résultats fournis (fichiers `.json` et `.csv`) :

- `experiments_pretrained/v2/cv/` : résultats et scores hors pli des sondes gelées ;
- `experiments_pretrained/v2/lcnn/` : résultats et scores du LCNN (sans les modèles entraînés) ;
- `experiments_pretrained/v2/<modèle>/index.csv` : ordre des fichiers des représentations extraites ;
- `experiments_pretrained/equalized/cv_speakers_grp*/` : contrôle de la fuite de voix ;
- `audit/*.json`, `audit/*.out` : contrôles et sorties des scripts.

Les représentations extraites (`mean.npy`, `std.npy`), les empreintes de voix et les modèles entraînés ne sont pas fournis : ils sont recalculés par les scripts.

## Reconstruire la base et les résultats

1. Installer l'environnement (Python 3.11) : `pip install -r requirements.txt`.
2. Télécharger le corpus Kallaama (wolof) et le placer dans `speech_dataset/`.
3. Obtenir les modèles de génération :
   - XTTS v2 adapté au wolof (GalsenAI, accès conditionné sur Hugging Face) : définir la variable d'environnement `HF_TOKEN` ;
   - SpeechT5 wolof : `bilalfaye/speecht5_tts-wolof` ;
   - kNN-VC : `bshall/knn-vc`.
4. Exécuter les étapes du tableau ci-dessus dans l'ordre.

Les scripts se lancent depuis la racine du dépôt.

## Licences et usage

- Corpus Kallaama : CC BY 4.0.
- Les modèles utilisés gardent leurs licences propres. XTTS v2 est sous Coqui Public Model License (usage non commercial). Le modèle wolof de GalsenAI est à accès conditionné.
- Les faux générés ne doivent pas être diffusés : ce sont des clones de voix de personnes réelles.
