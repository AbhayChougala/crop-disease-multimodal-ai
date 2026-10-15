from typing import Literal

ArchitectureName = Literal["resnet50", "efficientnet_b0", "mobilenet_v2"]
SUPPORTED_ARCHITECTURES: tuple[ArchitectureName, ...] = (
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v2",
)
