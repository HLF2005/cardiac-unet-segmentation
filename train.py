"""Train the U-Net baseline on the cardiac MRI dataset."""

import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EXPERIMENT_NAME,
    EXPERIMENTS_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAIN_PATIENTS,
    VALID_PATIENTS,
)
from src.data.dataset import CardiacDataset
from src.data.splits import create_patient_split
from src.models.unet import UNet
from src.training.experiment import (
    save_curves,
    save_history,
    save_json,
)
from src.training.trainer import validate_one_epoch
from src.training.losses import DiceCrossEntropyLoss


def main():
    torch.manual_seed(RANDOM_SEED) # Set random seed for reproducibility

    # Create output directory
    experiment_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Split the patient paths in train / validation / test sets
    training_path, valid_path, test_path = create_patient_split(
        data_dir=DATA_DIR,
        seed=RANDOM_SEED,
        train_patients=TRAIN_PATIENTS,
        valid_patients=VALID_PATIENTS,
    )

    # Create datasets
    training_dataset = CardiacDataset(training_path, augmentation=True)
    valid_dataset = CardiacDataset(valid_path, augmentation=False)
    test_dataset = CardiacDataset(test_path, augmentation=False)

    # Tests
    assert not set(training_path) & set(valid_path)
    assert not set(training_path) & set(test_path)
    assert not set(valid_path) & set(test_path)

    print(
        "Train      : "
        f"{len(training_dataset.patient_dirs)} patients, "
        f"{len(training_dataset)} coupes"
    )
    print(
        "Validation : "
        f"{len(valid_dataset.patient_dirs)} patients, "
        f"{len(valid_dataset)} coupes"
    )
    print(
        "Test       : "
        f"{len(test_dataset.patient_dirs)} patients, "
        f"{len(test_dataset)} coupes"
    )

    # Build data loader
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device used : {device}")

    model = UNet(in_channels=1, out_channels=NUM_CLASSES).to(device)
    loss_function  = DiceCrossEntropyLoss(num_classes=NUM_CLASSES,
                                            ce_weight=1.0,
                                            dice_weight=1.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    experiment_config = {
        "experiment_name": EXPERIMENT_NAME,
        "loss": loss_function.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_classes": NUM_CLASSES,
        "seed": RANDOM_SEED,
        "train_patients": TRAIN_PATIENTS,
        "valid_patients": VALID_PATIENTS,
        "augmentation": training_dataset.augmentation,

        "scheduler": scheduler.__class__.__name__,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
    }
    save_json(experiment_config, experiment_dir / "config.json")

    print(f"Experiments : {EXPERIMENT_NAME}")
    print(f"Result : {experiment_dir}")

    history = []
    best_dice = float("-inf")

    #  Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for images, targets in training_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            train_loss = loss_function(logits, targets)

            train_loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_train_loss += train_loss.item() * batch_size
            total_train_samples += batch_size

        average_train_loss = total_train_loss / total_train_samples

        # Validation
        metrics = validate_one_epoch(
            model=model,
            data_loader=valid_loader,
            loss_function=loss_function,
            device=device,
            num_classes=NUM_CLASSES,
        )

        average_val_loss = metrics["loss"]
        average_dice = metrics["dice_mean"]
        dice_per_class = metrics["dice_per_class"]
        current_learning_rate = optimizer.param_groups[0]["lr"]

        scheduler.step(average_dice)

        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
            "val_dice": average_dice,
            "dice_class_1": dice_per_class[0],
            "dice_class_2": dice_per_class[1],
            "dice_class_3": dice_per_class[2],
            "learning_rate": current_learning_rate,
        }
        history.append(epoch_results)
        save_history( history, experiment_dir / "history.csv")

        if average_dice > best_dice:
            best_dice = average_dice
            torch.save(model.state_dict(),experiment_dir / "best_model.pth")
            save_json(
                {
                    "best_epoch": epoch + 1,
                    "best_val_loss": average_val_loss,
                    "best_val_dice": average_dice,
                    "dice_class_1": dice_per_class[0],
                    "dice_class_2": dice_per_class[1],
                    "dice_class_3": dice_per_class[2],
                },
                experiment_dir / "summary.json",
            )

        dice_details = " | ".join(
            f"Dice classe {class_idx}: {dice:.4f}"
            for class_idx, dice in enumerate(
                dice_per_class,
                start=1,
            )
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"| LR: {current_learning_rate:.2e} "
            f"| Train loss: {average_train_loss:.4f} "
            f"| Val loss: {average_val_loss:.4f} "
            f"| Val Dice: {average_dice:.4f} "
            f"| {dice_details}"
        )

    save_curves( history, experiment_dir / "curves.png")


if __name__ == "__main__":
    main()
