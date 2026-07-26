"""Verify that the model can memorize four training slices."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import DATA_DIR, NUM_CLASSES
from src.data.dataset import CardiacDataset
from src.metrics.segmentation import dice_score
from src.models.unet import UNet


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_dataset = CardiacDataset(
        patient_dirs=[DATA_DIR / "patient001"]
    )
    train_subset = Subset(
        train_dataset,
        indices=[1, 2, 3, 4],
    )
    training_loader = DataLoader(
        train_subset,
        batch_size=4,
        num_workers=0,
    )

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 50
    train_losses = []
    dices = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_dice = 0.0
        model.train()

        for images, targets in training_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            train_loss = loss_function(logits, targets)

            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()

            predictions = torch.argmax(
                logits.detach(),
                dim=1,
            )
            dice = dice_score(
                predictions,
                targets,
                num_classes=NUM_CLASSES,
            )
            total_dice += dice.item()

        average_loss = total_loss / len(training_loader)
        average_dice = total_dice / len(training_loader)

        train_losses.append(average_loss)
        dices.append(average_dice)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"| Train: {average_loss:.4f} "
            f"| Dice: {average_dice:.4f}"
        )

    model.eval()
    with torch.no_grad():
        for images, targets in training_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            dice = dice_score(
                predictions,
                targets,
                num_classes=NUM_CLASSES,
            )
            print(f"Dice total : {dice.item():.4f}")


if __name__ == "__main__":
    main()
