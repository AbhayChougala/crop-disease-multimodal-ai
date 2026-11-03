from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from crop_ai.models.checkpoints import load_model_from_checkpoint


def test_legacy_efficientnet_checkpoint_loads() -> None:
    checkpoint = Path(__file__).parents[2] / "model" / "best_model.pth"
    model, labels, metadata, transform = load_model_from_checkpoint(
        checkpoint, torch.device("cpu")
    )

    assert metadata.architecture == "efficientnet_b0"
    assert len(labels) == 38
    assert transform.image_size == 224
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 38)
