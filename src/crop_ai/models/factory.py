from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision import models, transforms

from crop_ai.models.registry import SUPPORTED_ARCHITECTURES, ArchitectureName


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def as_dict(self) -> dict[str, object]:
        return {"image_size": self.image_size, "mean": list(self.mean), "std": list(self.std)}


def create_model(
    architecture: ArchitectureName,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
    unfreeze_blocks: int = 0,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("Classification requires at least two classes.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    elif architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Expected one of {SUPPORTED_ARCHITECTURES}."
        )

    configure_trainable_layers(model, architecture, freeze_backbone, unfreeze_blocks)
    return model


def _head(model: nn.Module, architecture: ArchitectureName) -> nn.Module:
    return model.fc if architecture == "resnet50" else model.classifier


def _feature_blocks(model: nn.Module, architecture: ArchitectureName) -> list[nn.Module]:
    if architecture == "resnet50":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    return list(model.features.children())


def configure_trainable_layers(
    model: nn.Module,
    architecture: ArchitectureName,
    freeze_backbone: bool,
    unfreeze_blocks: int = 0,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not freeze_backbone
    if not freeze_backbone:
        return
    for parameter in _head(model, architecture).parameters():
        parameter.requires_grad = True
    if unfreeze_blocks > 0:
        for block in _feature_blocks(model, architecture)[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def get_target_layer(model: nn.Module, architecture: ArchitectureName) -> nn.Module:
    if architecture == "resnet50":
        return model.layer4[-1].conv3
    if architecture in {"efficientnet_b0", "mobilenet_v2"}:
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer is registered for '{architecture}'.")


def build_inference_transform(config: TransformConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
