from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - ((2 * intersection + epsilon) / (denominator + epsilon))).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits, targets)
        return (1 - self.dice_weight) * bce + self.dice_weight * soft_dice_loss(logits, targets)
