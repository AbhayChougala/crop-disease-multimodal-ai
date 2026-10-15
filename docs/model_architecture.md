# Model architecture

The registry supports `resnet50`, `efficientnet_b0`, and `mobilenet_v2` using maintained torchvision
implementations. Each replaces the final classification head, supports dropout and backbone freezing,
and registers a compatible final convolution layer for Grad-CAM. Tests always use `weights=None`, so
they require no network download.

Runtime flow:

1. `ModelManager` loads one configured checkpoint at startup/on first access.
2. `TorchClassifier` applies the transform embedded in checkpoint metadata under inference mode.
3. It returns raw/calibrated confidence, ordered top-k classes, model/checkpoint provenance, latency,
   and an explicit abstention below the configured threshold.
4. Grad-CAM temporarily enables gradients for the selected class and removes hooks after use.

`MODEL_MODE=real` never falls back to demo predictions. An absent or incompatible checkpoint changes
the runtime state to `UNAVAILABLE` and diagnosis returns a service error. `MODEL_MODE=mock` is an
explicit deterministic UI mode and is not research evidence.

The U-Net module is separate from classification. A trained segmentation checkpoint is unusable by
the runtime until held-out annotated-mask evaluation has explicitly marked it validated.
