from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from crop_ai.explainability.gradcam import GradCAM
from crop_ai.models.checkpoints import (
    CheckpointMetadata,
    load_model_from_checkpoint,
    save_checkpoint,
)
from crop_ai.models.classifier import TorchClassifier
from crop_ai.models.contracts import ClassLabel
from crop_ai.models.factory import TransformConfig, create_model, get_target_layer
from crop_ai.models.registry import SUPPORTED_ARCHITECTURES


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_factory_forward_dimensions_without_weights(architecture: str) -> None:
    model = create_model(architecture, 3, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 3)


def _checkpoint(path: Path) -> None:
    model = create_model("mobilenet_v2", 2, pretrained=False)
    labels = {
        0: ClassLabel(0, "tomato__healthy", "Tomato", "Healthy"),
        1: ClassLabel(1, "tomato__early_blight", "Tomato", "Early Blight"),
    }
    metadata = CheckpointMetadata(
        architecture="mobilenet_v2",
        model_version="test",
        training_dataset="synthetic",
        dataset_fingerprint="fixture",
        seed=7,
        epoch=1,
        best_metric=None,
        transform=TransformConfig(image_size=64).as_dict(),
    )
    save_checkpoint(path, model, labels, metadata)


def test_training_batch_backprop_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = create_model("mobilenet_v2", 2, pretrained=False)
    model.train()
    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    checkpoint = tmp_path / "toy.pt"
    _checkpoint(checkpoint)
    restored, mapping, metadata, _ = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    assert restored(torch.randn(1, 3, 64, 64)).shape == (1, 2)
    assert mapping[1].class_id == "tomato__early_blight"
    assert metadata.training_dataset == "synthetic"


def test_top_k_abstention_and_gradcam(tmp_path: Path) -> None:
    checkpoint = tmp_path / "toy.pt"
    _checkpoint(checkpoint)
    classifier = TorchClassifier.from_checkpoint(
        checkpoint, torch.device("cpu"), confidence_threshold=1.0
    )
    prediction = classifier.predict(np.full((64, 64, 3), 120, dtype=np.uint8), top_k=2)
    assert len(prediction.top_k) == 2
    assert prediction.abstained is True
    assert prediction.crop is None

    model = classifier.model
    tensor = classifier._tensor(np.full((64, 64, 3), 120, dtype=np.uint8))
    with GradCAM(model, get_target_layer(model, "mobilenet_v2")) as generator:
        result = generator.generate(tensor, is_untrained_demo=True)
    assert result.heatmap.shape == (64, 64)
    assert np.isfinite(result.heatmap).all()
    assert result.is_untrained_demo is True
