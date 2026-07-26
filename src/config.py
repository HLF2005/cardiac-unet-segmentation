"""Shared project configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "training"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

EXPERIMENT_NAME = "ce_dice_25epochs"

RANDOM_SEED = 42
NUM_CLASSES = 4
TRAIN_PATIENTS = 70
VALID_PATIENTS = 15

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
