"""Evaluate the final model separately for each test patient."""

import csv

import numpy as np
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
from src.training.losses import DiceCrossEntropyLoss
from src.training.trainer import validate_one_epoch


def main():
    experiment_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME

    _, _, test_path = create_patient_split(
        data_dir=DATA_DIR,
        seed=RANDOM_SEED,
        train_patients=TRAIN_PATIENTS,
        valid_patients=VALID_PATIENTS,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
    ).to(device)
    model.load_state_dict(
        torch.load(
            experiment_dir / "best_model.pth",
            map_location=device,
        )
    )

    loss_function = DiceCrossEntropyLoss(
        num_classes=NUM_CLASSES,
    )

    patient_results = []

    for patient_dir in test_path:
        patient_dataset = CardiacDataset([patient_dir])
        patient_loader = DataLoader(
            patient_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        metrics = validate_one_epoch(
            model=model,
            data_loader=patient_loader,
            loss_function=loss_function,
            device=device,
            num_classes=NUM_CLASSES,
        )

        dice_per_class = metrics["dice_per_class"]
        result = {
            "patient": patient_dir.name,
            "slices": len(patient_dataset),
            "dice_mean": metrics["dice_mean"],
            "dice_class_1": dice_per_class[0],
            "dice_class_2": dice_per_class[1],
            "dice_class_3": dice_per_class[2],
        }
        patient_results.append(result)

        print(
            f"{patient_dir.name} "
            f"| Dice: {metrics['dice_mean']:.4f}"
        )

    output_path = experiment_dir / "patient_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=patient_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(patient_results)

    patient_dices = np.array(
        [row["dice_mean"] for row in patient_results]
    )
    best_patient = max(
        patient_results,
        key=lambda row: row["dice_mean"],
    )
    worst_patient = min(
        patient_results,
        key=lambda row: row["dice_mean"],
    )

    summary = {
        "patients": len(patient_results),
        "mean_dice": patient_dices.mean().item(),
        "std_dice": patient_dices.std().item(),
        "min_dice": patient_dices.min().item(),
        "max_dice": patient_dices.max().item(),
        "best_patient": best_patient["patient"],
        "worst_patient": worst_patient["patient"],
    }

    for class_idx in range(1, NUM_CLASSES):
        class_dices = np.array(
            [
                row[f"dice_class_{class_idx}"]
                for row in patient_results
            ]
        )
        summary[f"class_{class_idx}_mean"] = (
            class_dices.mean().item()
        )
        summary[f"class_{class_idx}_std"] = (
            class_dices.std().item()
        )

    save_json(
        summary,
        experiment_dir / "patient_summary.json",
    )

    print(
        "Dice par patient : "
        f"{summary['mean_dice']:.4f} "
        f"± {summary['std_dice']:.4f}"
    )
    print(
        f"Pire patient : {summary['worst_patient']} "
        f"({summary['min_dice']:.4f})"
    )
    print(
        f"Meilleur patient : {summary['best_patient']} "
        f"({summary['max_dice']:.4f})"
    )


if __name__ == "__main__":
    main()
