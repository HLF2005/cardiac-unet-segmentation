import torch


def validate_one_epoch(model, data_loader, loss_function, device, num_classes):
    model.eval()

    total_val_loss = 0.0
    total_samples = 0

    intersections = torch.zeros(num_classes, device=device)
    denominators = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for X_val, y_val in data_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            logits = model(X_val)
            loss = loss_function(logits, y_val)

            batch_size = X_val.size(0)
            total_val_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)

            for class_idx in range(1, num_classes):
                pred_class = predictions == class_idx
                target_class = y_val == class_idx

                intersections[class_idx] += (
                    pred_class & target_class
                ).sum()

                denominators[class_idx] += (
                    pred_class.sum() + target_class.sum()
                )

    dice_per_class = []

    for class_idx in range(1, num_classes):
        if denominators[class_idx].item() > 0:
            dice = (
                2 * intersections[class_idx]
                / denominators[class_idx]
            )
            dice_per_class.append(dice.item())

    average_val_loss = total_val_loss / total_samples
    dice_mean = sum(dice_per_class) / len(dice_per_class)

    return {
        "loss": average_val_loss,
        "dice_mean": dice_mean,
        "dice_per_class": dice_per_class,
    }