from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a usable image."""


def decode_image(data: bytes, max_pixels: int = 40_000_000) -> np.ndarray:
    if not data:
        raise ImageDecodeError("The uploaded image is empty.")
    try:
        with Image.open(BytesIO(data)) as header:
            width, height = header.size
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError("The uploaded file is not a supported image.") from exc
    if min(width, height) < 32:
        raise ImageDecodeError("Image dimensions must be at least 32 x 32 pixels.")
    if width * height > max_pixels:
        raise ImageDecodeError(f"Image exceeds the {max_pixels:,}-pixel safety limit.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ImageDecodeError("The uploaded file could not be decoded as an RGB image.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageDecodeError("The processed image could not be encoded.")
    return encoded.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ImageDecodeError("The processed image could not be encoded.")
    return encoded.tobytes()
