from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LeafSegmentation:
    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    foreground_percent: float
    is_valid: bool
    warning: str | None = None


def segment_leaf(image_bgr: np.ndarray) -> LeafSegmentation:
    """Heuristic vegetation mask for simple and medium-complexity backgrounds."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    green_hsv = cv2.inRange(hsv, np.array([20, 25, 20]), np.array([105, 255, 255]))
    a_channel = lab[:, :, 1]
    green_lab = cv2.inRange(a_channel, 0, 132)
    mask = cv2.bitwise_and(green_hsv, green_lab)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return LeafSegmentation(mask, None, 0.0, False, "No plausible leaf region was found.")

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    pixels = int(cv2.countNonZero(clean))
    foreground = 100.0 * pixels / clean.size
    bbox = tuple(int(value) for value in cv2.boundingRect(largest))

    if foreground < 2.0:
        return LeafSegmentation(
            clean,
            bbox,
            round(foreground, 2),
            False,
            "Detected leaf region is too small for reliable severity estimation.",
        )
    if foreground > 95.0:
        return LeafSegmentation(
            clean,
            bbox,
            round(foreground, 2),
            False,
            "Detected foreground covers nearly the entire image; background separation "
            "is unreliable.",
        )
    return LeafSegmentation(clean, bbox, round(foreground, 2), True)
