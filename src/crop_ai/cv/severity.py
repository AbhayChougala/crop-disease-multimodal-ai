from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SeverityResult:
    method: str
    affected_area_percent: float | None
    category: str
    is_experimental: bool = True


DEFAULT_THRESHOLDS = ((5.0, "Low"), (20.0, "Moderate"), (40.0, "High"))


def calculate_severity(
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    thresholds: tuple[tuple[float, str], ...] = DEFAULT_THRESHOLDS,
) -> SeverityResult:
    leaf_pixels = int(cv2.countNonZero(leaf_mask))
    if leaf_pixels == 0:
        return SeverityResult("experimental_color_segmentation", None, "Unavailable")

    lesions_inside_leaf = cv2.bitwise_and(lesion_mask, leaf_mask)
    affected = 100.0 * cv2.countNonZero(lesions_inside_leaf) / leaf_pixels
    category = "Very high"
    for upper_bound, label in thresholds:
        if affected < upper_bound:
            category = label
            break
    return SeverityResult(
        "experimental_color_segmentation",
        round(float(affected), 2),
        category,
    )
