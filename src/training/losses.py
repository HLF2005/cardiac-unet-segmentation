import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)

        # Transform logit to probabilities
        probabilities = F.softmax(logits, dim=1)

        # One-hot encoding for the label
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute( 0, 3, 1, 2).float() # dim format: [Batch, Classes, H, W].

        # Skip the class background (classe 0)
        probabilities = probabilities[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

        # Calcul of th Dice Loss
        dimensions = (0, 2, 3)
        intersection = ( probabilities * targets_one_hot).sum(dim=dimensions)

        denominator = (probabilities.sum(dim=dimensions) + targets_one_hot.sum(dim=dimensions))

        dice_per_class = ( 2 * intersection + self.smooth) / ( denominator + self.smooth)

        dice_loss = 1 - dice_per_class.mean()

        # Total loss
        total_loss = ( self.ce_weight * ce_loss + self.dice_weight * dice_loss)

        return total_loss
