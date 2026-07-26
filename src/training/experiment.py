"""Utilities for saving experiment results."""

import csv
import json

import matplotlib.pyplot as plt


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_history(history, path):
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)


def save_curves(history, path):
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    val_dices = [row["val_dice"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Évolution de la loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_dices, label="Validation Dice")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Évolution du Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
