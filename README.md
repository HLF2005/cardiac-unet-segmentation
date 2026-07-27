# Segmentation cardiaque avec U-Net

Projet de segmentation 2D d'IRM cardiaques avec un U-Net.

Le modèle prédit quatre classes pour chaque pixel : le fond et trois
structures cardiaques.

## Organisation

```text
.
├── data/
│   └── training/                 # Volumes et masques NIfTI
├── outputs/
│   └── experiments/              # Un dossier par expérience
├── scripts/
│   ├── data/                     # Inspection du dataset
│   ├── evaluation/               # Visualisation des prédictions
│   └── training/                 # Tests du pipeline d'entraînement
├── src/
│   ├── data/                     # Dataset
│   ├── metrics/                  # Métriques de segmentation
│   ├── models/                   # Architecture U-Net
│   ├── training/                 # Validation
│   └── config.py                 # Configuration 
└── train.py                      # Entraînement principal
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commandes

Toutes les commandes sont à lancer depuis la racine du projet.

Inspecter tout le dataset :

```bash
python -m scripts.data.inspect_dataset
```

Entraîner le modèle :

```bash
python train.py
```

Visualiser les prédictions du meilleur checkpoint :

```bash
python -m scripts.evaluation.visualize_predictions
```

Évaluer une seule fois le modèle final sur le test set :

```bash
python -m scripts.evaluation.evaluate_test
```

Comparer automatiquement les expériences :

```bash
python -m scripts.evaluation.compare_experiments
```

Calculer les résultats séparément pour chaque patient :

```bash
python -m scripts.evaluation.evaluate_per_patient
```

Créer les superpositions pour la présentation :

```bash
python -m scripts.evaluation.visualize_overlays
```

Lancer les tests :

```bash
python -m unittest discover -s tests
```

## Split
Le dataset peut être récupéré avec ce lien: https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html

Le split est réalisé au niveau patient afin d'éviter qu'un même patient
apparaisse dans plusieurs ensembles :

- 70 patients pour l'entraînement 
- 15 patients pour la validation 
- 15 patients pour le test 

Ces valeurs et les principaux hyperparamètres se trouvent dans
`src/config.py`.

## Suivi des expériences

Le nom de l'expérience active est défini avec `EXPERIMENT_NAME` dans
`src/config.py`. Chaque expérience enregistre automatiquement :

- `config.json` : hyperparamètres ;
- `history.csv` : métriques de chaque époque ;
- `summary.json` : meilleure époque de validation ;
- `curves.png` : courbes de loss et de Dice ;
- `best_model.pth` : meilleur checkpoint ;
- `test_metrics.json` : résultats de test, seulement si le script de test
  est lancé.

## Expériences

| Expérience | Configuration principale | Dice validation |
|---|---|---:|
| `baseline_ce` | Cross-Entropy, 10 époques, batch 8 | 0.8742 |
| `ce_dice` | Cross-Entropy + Dice, 10 époques, batch 8 | 0.8913 |
| `ce_dice_25epochs` | Cross-Entropy + Dice, 25 époques, batch 8 | **0.9157** |
| `ce_dice_scheduler` | Scheduler, 30 époques, batch 16 | 0.9110 |
| `ce_dice_augmentation_only` | Augmentation, 25 époques, batch 8 | 0.9125 |
| `ce_dice_augmentation_30epochs` | Augmentation, 30 époques, batch 8 | 0.9155 |
| `ce_dice_augmentation` | Augmentation + scheduler, 30 époques, batch 16 | 0.9125 |

Le modèle final est `ce_dice_25epochs`, sélectionné uniquement avec les
résultats de validation.

### Analyse des expériences

L'ajout de la Dice Loss à la Cross-Entropy améliore le Dice de validation
de `0.8742` à `0.8913`. Le gain principal concerne la classe 2 :

| Métrique | Cross-Entropy | CE + Dice | Gain |
|---|---:|---:|---:|
| Dice moyen | 0.8742 | 0.8913 | +0.0171 |
| Classe 1 | 0.8684 | 0.8823 | +0.0139 |
| Classe 2 | 0.8237 | 0.8519 | **+0.0282** |
| Classe 3 | 0.9305 | 0.9396 | +0.0091 |

Prolonger l'entraînement de 10 à 25 époques produit un second gain
important :

| Métrique | 10 époques | 25 époques | Gain |
|---|---:|---:|---:|
| Dice moyen | 0.8913 | **0.9157** | +0.0245 |
| Classe 1 | 0.8823 | 0.9161 | +0.0338 |
| Classe 2 | 0.8519 | 0.8768 | +0.0248 |
| Classe 3 | 0.9396 | 0.9543 | +0.0147 |

Le meilleur checkpoint est obtenu à l'époque 21. Les performances
atteignent ensuite un plateau autour de `0.91`.

Le scheduler stabilise la fin de l'entraînement, mais ne dépasse pas le
meilleur modèle. Son effet ne peut pas être complètement isolé car cette
expérience utilise aussi un batch de 16.

L'augmentation seule atteint `0.9125` après 25 époques et `0.9155` après
30 époques. Elle ne produit donc pas de gain mesurable par rapport au
modèle sans augmentation (`0.9157`).

Les valeurs de loss ne sont pas comparées entre la baseline et les
autres expériences, car elles correspondent à des fonctions différentes.

## Résultats finaux

Le modèle final atteint les résultats suivants sur les 15 patients du
test set :

| Métrique | Dice |
|---|---:|
| Moyenne des trois classes | **0.9163** |
| Classe 1 | 0.9035 |
| Classe 2 | 0.8863 |
| Classe 3 | 0.9591 |

Sur les 15 patients du test, le Dice moyen calculé séparément par
patient est :

```text
0.9023 ± 0.0516
```

Le meilleur patient obtient `0.9549` et le cas le plus difficile
`0.7619` :

```text
Meilleur patient : patient078
Pire patient     : patient037
```

| Dice par patient | Moyenne ± écart-type |
|---|---:|
| Classe 1 | 0.8811 ± 0.0962 |
| Classe 2 | 0.8784 ± 0.0399 |
| Classe 3 | 0.9473 ± 0.0362 |

Les fichiers détaillés et les courbes sont disponibles dans
`outputs/experiments/`. 

Le Dice test global est presque identique au Dice de validation :

```text
Validation : 0.9157
Test       : 0.9163
Écart      : +0.0006
```

Cette proximité indique que le modèle généralise correctement sur les
patients non vus. La classe 3 est la plus facile à segmenter. La classe 1
présente la plus forte variabilité entre patients.

### Comparaison des expériences

![Comparaison des expériences](outputs/results/experiment_comparison.png)

### Exemples de segmentation

Les couleurs représentent les trois classes cardiaques. La colonne
centrale montre la vérité terrain et la colonne de droite la prédiction.

![Masques superposés sur les IRM](outputs/experiments/ce_dice_25epochs/test_overlays.png)

## Données

Les données médicales ne sont pas incluses dans le repo. Elles doivent
être placées dans :

```text
data/training/patientXXX/
```

Chaque volume NIfTI doit être accompagné de son masque dont le nom se
termine par `_gt.nii.gz`.

## Limites

- **Attention U-Net** : ajouter des attention gates pour que le modèle se concentre sur les zones cardiaques et ignore le fond
- **U-Net 3D** : exploiter la dimension spatiale des volumes IRM plutôt que de traiter les slices 2D indépendamment
- **Pre-training** : partir d'un encodeur pré-entraîné sur ImageNe

