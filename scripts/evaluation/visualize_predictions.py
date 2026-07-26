"""Visualize validation images, ground truths, and model predictions."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from src.config import (
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


def main():
    experiment_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME

    _, valid_path, _ = create_patient_split(
        data_dir=DATA_DIR,
        seed=RANDOM_SEED,
        train_patients=TRAIN_PATIENTS,
        valid_patients=VALID_PATIENTS,
    )
    valid_dataset = CardiacDataset(valid_path)

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
    state_dict = torch.load(
        experiment_dir / "best_model.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    sample_indices = np.linspace(
        0,
        len(valid_dataset) - 1,
        num=4,
        dtype=int,
    )

    mask_cmap = ListedColormap(
        ["black", "red", "green", "blue"]
    )
    figure, axes = plt.subplots(
        nrows=len(sample_indices),
        ncols=3,
        figsize=(12, 4 * len(sample_indices)),
    )

    with torch.no_grad():
        for row, index in enumerate(sample_indices):
            image, target = valid_dataset[index]

            logits = model(image.unsqueeze(0).to(device))
            prediction = torch.argmax(
                logits,
                dim=1,
            ).squeeze(0).cpu()

            axes[row, 0].imshow(image.squeeze(0), cmap="gray")
            axes[row, 0].set_title(f"IRM — coupe {index}")

            axes[row, 1].imshow(
                target,
                cmap=mask_cmap,
                vmin=0,
                vmax=NUM_CLASSES - 1,
                interpolation="nearest",
            )
            axes[row, 1].set_title("Masque réel")

            axes[row, 2].imshow(
                prediction,
                cmap=mask_cmap,
                vmin=0,
                vmax=NUM_CLASSES - 1,
                interpolation="nearest",
            )
            axes[row, 2].set_title("Prédiction")

            for axis in axes[row]:
                axis.axis("off")

    figure.tight_layout()

    output_path = experiment_dir / "validation_predictions.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Figure sauvegardée dans : {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
