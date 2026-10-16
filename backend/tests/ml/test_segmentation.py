import pytest

torch = pytest.importorskip("torch")

from crop_ai.segmentation.metrics import segmentation_metrics
from crop_ai.segmentation.unet import UNet


def test_unet_mask_shape_and_metrics() -> None:
    model = UNet(base=4)
    logits = model(torch.randn(1, 3, 65, 67))
    assert logits.shape == (1, 1, 65, 67)

    values = segmentation_metrics(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    assert values["dice"] == pytest.approx(1.0)
    assert values["iou"] == pytest.approx(1.0)
