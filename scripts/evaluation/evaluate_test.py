"""Evaluate the best model on the test dataset."""

import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EXPERIMENT_NAME,
    EXPERIMENTS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    TRAIN_PATIENTS,
    VALID_PATIENTS,
)
from src.data.dataset import CardiacDataset
from src.data.splits import create_patient_split
from src.models.unet import UNet
from src.training.experiment import save_json
from src.training.trainer import validate_one_epoch
from src.training.losses import DiceCrossEntropyLoss

def main():
    experiment_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME

    _, _, test_path = create_patient_split(
        data_dir=DATA_DIR,
        seed=RANDOM_SEED,
        train_patients=TRAIN_PATIENTS,
        valid_patients=VALID_PATIENTS,
    )

    test_dataset = CardiacDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device utilisé : {device}")
    print(
        f"Test : {len(test_dataset.patient_dirs)} patients, "
        f"{len(test_dataset)} coupes"
    )

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
    ).to(device)

    checkpoint_path = experiment_dir / "best_model.pth"
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    loss_function = DiceCrossEntropyLoss(
        num_classes=NUM_CLASSES,
        ce_weight=1.0,
        dice_weight=1.0,
    )

    metrics = validate_one_epoch(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        num_classes=NUM_CLASSES,
    )

    test_loss = metrics["loss"]
    test_dice = metrics["dice_mean"]
    dice_per_class = metrics["dice_per_class"]

    print(f"Test loss : {test_loss:.4f}")
    print(f"Test Dice moyen : {test_dice:.4f}")

    for class_idx, dice in enumerate(dice_per_class, start=1):
        print(f"Test Dice classe {class_idx} : {dice:.4f}")

    save_json(
        {
            "test_loss": test_loss,
            "test_dice": test_dice,
            "dice_class_1": dice_per_class[0],
            "dice_class_2": dice_per_class[1],
            "dice_class_3": dice_per_class[2],
        },
        experiment_dir / "test_metrics.json",
    )


if __name__ == "__main__":
    main()
