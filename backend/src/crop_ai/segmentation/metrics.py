from __future__ import annotations

import torch


def segmentation_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = targets >= 0.5
    intersection = (predictions & truth).sum().item()
    union = (predictions | truth).sum().item()
    predicted = predictions.sum().item()
    actual = truth.sum().item()
    return {
        "iou": float((intersection + epsilon) / (union + epsilon)),
        "dice": float((2 * intersection + epsilon) / (predicted + actual + epsilon)),
        "pixel_precision": float((intersection + epsilon) / (predicted + epsilon)),
        "pixel_recall": float((intersection + epsilon) / (actual + epsilon)),
    }
