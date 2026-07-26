"""Overlay ground-truth and predicted masks on the original MRI."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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

    _, _, test_path = create_patient_split(
        data_dir=DATA_DIR,
        seed=RANDOM_SEED,
        train_patients=TRAIN_PATIENTS,
        valid_patients=VALID_PATIENTS,
    )
    test_dataset = CardiacDataset(test_path)

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
    model.eval()

    # Une coupe riche en structures dans chaque quart du test set.
    selected_indices = []
    sections = np.array_split(
        np.arange(len(test_dataset)),
        4,
    )

    for section in sections:
        best_index = max(
            section,
            key=lambda index: torch.count_nonzero(
                test_dataset[int(index)][1]
            ).item(),
        )
        selected_indices.append(int(best_index))

    mask_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.55),
            (0.0, 1.0, 0.0, 0.55),
            (0.0, 0.4, 1.0, 0.55),
        ]
    )

    figure, axes = plt.subplots(
        len(selected_indices),
        3,
        figsize=(12, 4 * len(selected_indices)),
    )

    with torch.no_grad():
        for row, index in enumerate(selected_indices):
            image, target = test_dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            prediction = torch.argmax(
                logits,
                dim=1,
            ).squeeze(0).cpu()

            image_2d = image.squeeze(0)

            axes[row, 0].imshow(image_2d, cmap="gray")
            axes[row, 0].set_title(f"IRM — coupe {index}")

            axes[row, 1].imshow(image_2d, cmap="gray")
            axes[row, 1].imshow(
                target,
                cmap=mask_cmap,
                vmin=0,
                vmax=NUM_CLASSES - 1,
                interpolation="nearest",
            )
            axes[row, 1].set_title("Masque réel superposé")

            axes[row, 2].imshow(image_2d, cmap="gray")
            axes[row, 2].imshow(
                prediction,
                cmap=mask_cmap,
                vmin=0,
                vmax=NUM_CLASSES - 1,
                interpolation="nearest",
            )
            axes[row, 2].set_title("Prédiction superposée")

            for axis in axes[row]:
                axis.axis("off")

    legend = [
        Patch(color="red", label="Classe 1"),
        Patch(color="lime", label="Classe 2"),
        Patch(color="royalblue", label="Classe 3"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
    )
    figure.suptitle(
        f"Segmentation du modèle {EXPERIMENT_NAME}",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))

    output_path = experiment_dir / "test_overlays.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Figure sauvegardée : {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
