from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    abstention_threshold: float | None = None,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro = precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    one_hot = np.eye(len(class_names))[targets]
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "expected_calibration_error": expected_calibration_error(probabilities, targets),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confidence_distribution": {
            "bin_edges": np.linspace(0, 1, 11).tolist(),
            "counts": np.histogram(probabilities.max(axis=1), bins=np.linspace(0, 1, 11))[
                0
            ].tolist(),
        },
    }
    try:
        if len(class_names) == 2:
            result["roc_auc"] = float(roc_auc_score(targets, probabilities[:, 1]))
        elif len(np.unique(targets)) == len(class_names):
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro")
            )
    except ValueError:
        result["roc_auc_status"] = "unavailable_for_present_classes"
    for top_k in (3, 5):
        if len(class_names) >= top_k:
            top = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
            result[f"top_{top_k}_accuracy"] = float(
                np.mean([target in row for target, row in zip(targets, top, strict=False)])
            )
    if len(class_names) == 2:
        result["binary_brier_score"] = float(brier_score_loss(targets, probabilities[:, 1]))
    if abstention_threshold is not None:
        accepted = probabilities.max(axis=1) >= abstention_threshold
        result["coverage"] = float(accepted.mean())
        result["selective_accuracy"] = (
            float((predictions[accepted] == targets[accepted]).mean()) if accepted.any() else None
        )
    return result
