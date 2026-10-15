from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class SeverityResult:
    method: str
    affected_area_percent: float | None
    category: str
    status: Literal["unavailable", "experimental", "validated"]
    is_experimental: bool = True
    warning: str | None = None


def calculate_severity(
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    thresholds: tuple[tuple[float, str], ...] | None = None,
    validated_masks: bool = False,
) -> SeverityResult:
    if leaf_mask.ndim != 2 or lesion_mask.ndim != 2 or leaf_mask.shape != lesion_mask.shape:
        return SeverityResult(
            "mask_area_ratio",
            None,
            "Unavailable",
            "unavailable",
            True,
            "Leaf and lesion masks must be two-dimensional and have identical shapes.",
        )
    leaf_pixels = int(cv2.countNonZero(leaf_mask))
    if leaf_pixels == 0:
        return SeverityResult(
            "mask_area_ratio",
            None,
            "Unavailable",
            "unavailable",
            True,
            "Severity requires a non-empty valid leaf mask.",
        )

    lesions_inside_leaf = cv2.bitwise_and(lesion_mask, leaf_mask)
    affected = 100.0 * cv2.countNonZero(lesions_inside_leaf) / leaf_pixels
    category = "Not categorized"
    if thresholds:
        category = thresholds[-1][1]
        for upper_bound, label in thresholds:
            if affected <= upper_bound:
                category = label
                break
    return SeverityResult(
        "validated_mask_area_ratio" if validated_masks else "experimental_color_segmentation",
        round(float(affected), 2),
        category,
        "validated" if validated_masks else "experimental",
        not validated_masks,
        None if validated_masks else "Color-based lesion candidates are an unvalidated baseline.",
    )
