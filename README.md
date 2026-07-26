# Segmentation cardiaque avec U-Net

Projet pédagogique de segmentation 2D d'IRM cardiaques avec un U-Net
implémenté en PyTorch.

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
│   ├── data/                     # Dataset et split patient
│   ├── metrics/                  # Métriques de segmentation
│   ├── models/                   # Architecture U-Net
│   ├── training/                 # Validation
│   └── config.py                 # Configuration partagée
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

Inspecter un exemple :

```bash
python -m scripts.data.inspect_sample
```

Tester le sur-apprentissage sur quatre coupes :

```bash
python -m scripts.training.overfit_small_batch
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

Calculer les résultats par patient :

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

Le split est réalisé au niveau patient afin d'éviter qu'un même patient
apparaisse dans plusieurs ensembles :

- 70 patients pour l'entraînement ;
- 15 patients pour la validation ;
- 15 patients pour le test ;
- graine aléatoire : `42`.

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
| `ce_dice_augmentation` | Augmentation + scheduler, 30 époques, batch 16 | 0.9125 |

Le modèle final est `ce_dice_25epochs`, sélectionné uniquement avec les
résultats de validation.

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
`0.7619`.

| Dice par patient | Moyenne ± écart-type |
|---|---:|
| Classe 1 | 0.8811 ± 0.0962 |
| Classe 2 | 0.8784 ± 0.0399 |
| Classe 3 | 0.9473 ± 0.0362 |

Les fichiers détaillés et les courbes sont disponibles dans
`outputs/experiments/`. Les checkpoints ne sont pas versionnés car ils
sont volumineux.

### Comparaison des expériences

![Comparaison des expériences](outputs/results/experiment_comparison.png)

### Exemples de segmentation

Les couleurs représentent les trois classes cardiaques. La colonne
centrale montre la vérité terrain et la colonne de droite la prédiction.

![Masques superposés sur les IRM](outputs/experiments/ce_dice_25epochs/test_overlays.png)

## Données

Les données médicales ne sont pas incluses dans le dépôt. Elles doivent
être placées dans :

```text
data/training/patientXXX/
```

Chaque volume NIfTI doit être accompagné de son masque dont le nom se
termine par `_gt.nii.gz`.

## Limites

- Le modèle traite les volumes sous forme de coupes 2D indépendantes.
- Le contexte entre les coupes voisines n'est pas utilisé.
- Les images sont redimensionnées à `256 × 256`.
- Les expériences avec scheduler utilisent aussi un batch size différent,
  ce qui limite l'interprétation isolée de cet hyperparamètre.
- Les résultats proviennent d'une seule graine d'entraînement.
