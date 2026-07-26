
import torch


def dice_score(predictions, targets, num_classes, background=False):
    if predictions.shape != targets.shape:
        raise ValueError
    if background:
        start_classe = 0
    else:
        start_classe = 1
    dice_scores = []
    for class_idx in range(start_classe, num_classes):
        mask_pred = predictions == class_idx
        mask_target = targets == class_idx

        intersection = (mask_pred & mask_target).sum()
        denominator = mask_pred.sum() + mask_target.sum()

        if denominator.item() == 0:
            continue
        dice = (2 * intersection.float()) / denominator.float()
        dice_scores.append(dice)


    if len(dice_scores) == 0:
        return torch.tensor(
            1.0,
            device=predictions.device,
            dtype=torch.float32,
        )

    return torch.stack(dice_scores).mean()



if __name__ == "__main__":
    targets = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])

    predictions = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])

    score = dice_score(
        predictions,
        targets,
        num_classes=2,
    )

    print("Recouvrement parfait :", score)

