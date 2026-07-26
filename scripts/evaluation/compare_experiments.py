"""Create a table and a figure comparing all experiments."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from src.config import EXPERIMENTS_DIR, OUTPUTS_DIR


def main():
    results = []

    for experiment_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        config_path = experiment_dir / "config.json"
        summary_path = experiment_dir / "summary.json"

        if not config_path.exists() or not summary_path.exists():
            continue

        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)

        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)

        results.append(
            {
                "experiment": experiment_dir.name,
                "loss": config["loss"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "augmentation": config["augmentation"],
                "scheduler": config.get("scheduler", "None"),
                "best_epoch": summary["best_epoch"],
                "val_dice": summary["best_val_dice"],
                "dice_class_1": summary["dice_class_1"],
                "dice_class_2": summary["dice_class_2"],
                "dice_class_3": summary["dice_class_3"],
            }
        )

    results.sort(key=lambda row: row["val_dice"])

    results_dir = OUTPUTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "experiment_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    names = [row["experiment"] for row in results]
    class_1 = [row["dice_class_1"] for row in results]
    class_2 = [row["dice_class_2"] for row in results]
    class_3 = [row["dice_class_3"] for row in results]
    positions = np.arange(len(results))
    width = 0.25

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions - width, class_1, width, label="Classe 1")
    axis.bar(positions, class_2, width, label="Classe 2")
    axis.bar(positions + width, class_3, width, label="Classe 3")

    axis.set_ylabel("Dice validation")
    axis.set_title("Comparaison des expériences")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0.75, 1.0)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure_path = results_dir / "experiment_comparison.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Tableau sauvegardé : {csv_path}")
    print(f"Figure sauvegardée : {figure_path}")


if __name__ == "__main__":
    main()
