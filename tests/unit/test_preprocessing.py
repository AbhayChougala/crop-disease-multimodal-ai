import cv2
import numpy as np

from crop_ai.cv.preprocessing import preprocess_image


def test_preprocessing_finds_simple_green_leaf() -> None:
    image = np.full((180, 240, 3), 245, dtype=np.uint8)
    cv2.ellipse(image, (120, 90), (70, 45), -20, 0, 360, (35, 145, 45), -1)

    result = preprocess_image(image, "standard")

    assert result.image.shape == image.shape
    assert result.leaf.is_valid
    assert result.leaf.foreground_percent > 5
    assert result.lesion_mask.shape == image.shape[:2]
