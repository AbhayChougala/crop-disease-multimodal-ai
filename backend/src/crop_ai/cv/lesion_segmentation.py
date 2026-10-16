from __future__ import annotations

import cv2
import numpy as np


def segment_lesions(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Experimental color-based baseline; this is not an annotated disease mask."""
    if cv2.countNonZero(leaf_mask) == 0:
        return np.zeros_like(leaf_mask)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    hue, saturation, value = cv2.split(hsv)
    _, a_channel, b_channel = cv2.split(lab)

    brown_or_yellow = (
        ((hue < 35) & (saturation > 55) & (value > 25))
        | ((a_channel > 140) & (b_channel > 135))
        | ((value < 70) & (saturation > 35))
    )
    lesion = np.where(brown_or_yellow, 255, 0).astype(np.uint8)
    lesion = cv2.bitwise_and(lesion, leaf_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)
    return lesion
