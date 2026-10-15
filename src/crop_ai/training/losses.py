from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probability = torch.exp(-cross_entropy)
        return (((1 - probability) ** self.gamma) * cross_entropy).mean()


def create_loss(
    name: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(focal_gamma, class_weights, label_smoothing)
    raise ValueError("loss must be 'cross_entropy' or 'focal'.")
