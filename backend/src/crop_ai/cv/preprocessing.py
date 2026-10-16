from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from crop_ai.cv.leaf_segmentation import LeafSegmentation, segment_leaf
from crop_ai.cv.lesion_segmentation import segment_lesions

PreprocessingMode = Literal["raw", "standard", "segmented"]


@dataclass(frozen=True)
class PreprocessingResult:
    image: np.ndarray
    leaf: LeafSegmentation
    lesion_mask: np.ndarray
    leaf_isolated: np.ndarray
    lesion_overlay: np.ndarray
    warnings: tuple[str, ...]


def resize_long_edge(image: np.ndarray, max_edge: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_image(
    image_bgr: np.ndarray,
    mode: PreprocessingMode = "standard",
) -> PreprocessingResult:
    resized = resize_long_edge(image_bgr)
    if mode == "raw":
        processed = resized.copy()
    elif mode in {"standard", "segmented"}:
        denoised = cv2.bilateralFilter(resized, 7, 45, 45)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lightness)
        processed = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    leaf = segment_leaf(processed)
    warnings: list[str] = []
    if leaf.warning:
        warnings.append(leaf.warning)
    lesion_mask = (
        segment_lesions(processed, leaf.mask) if leaf.is_valid else np.zeros_like(leaf.mask)
    )
    if mode == "segmented" and leaf.is_valid:
        processed = cv2.bitwise_and(processed, processed, mask=leaf.mask)
    leaf_isolated = cv2.bitwise_and(processed, processed, mask=leaf.mask)
    lesion_overlay = processed.copy()
    lesion_overlay[lesion_mask > 0] = (
        0.45 * lesion_overlay[lesion_mask > 0] + 0.55 * np.array([30, 70, 220])
    ).astype(np.uint8)
    return PreprocessingResult(
        processed,
        leaf,
        lesion_mask,
        leaf_isolated,
        lesion_overlay,
        tuple(warnings),
    )
