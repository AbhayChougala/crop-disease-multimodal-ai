import cv2
import numpy as np
import pytest

from crop_ai.cv.decode import ImageDecodeError, decode_image


def test_decode_valid_jpeg() -> None:
    source = np.full((64, 64, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok
    decoded = decode_image(encoded.tobytes())
    assert decoded.shape == source.shape


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_decode_rejects_invalid_bytes(payload: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(payload)
