# Training guide

Run the commands below from `backend/` with its virtual environment active.

Install the ML extra and verify the selected device:

```bash
python -m pip install -e '.[dev,ml]'
python scripts/check_device.py
```

Edit a YAML file under `configs/training/` to point at a prepared manifest and image root. The three
classification runs use the same saved split for a fair comparison:

```bash
python -m crop_ai.training.train --config configs/training/resnet50.yaml
python -m crop_ai.training.train --config configs/training/efficientnet_b0.yaml
python -m crop_ai.training.train --config configs/training/mobilenet_v2.yaml
```

Training supports ImageNet transfer learning, frozen/partially unfrozen backbones, AdamW or SGD,
cosine or plateau scheduling, early stopping, weighted loss or balanced sampling, gradient
accumulation/clipping, CUDA-only AMP, and resume from a safe state-dict checkpoint. Checkpoints store
the taxonomy, preprocessing, seed, dataset fingerprint, package versions, and training configuration.

Evaluate or predict with measured artifacts only:

```bash
python -m crop_ai.training.evaluate \
  --checkpoint artifacts/models/efficientnet_b0/best.pt \
  --manifest data/processed/plantvillage_manifest.csv \
  --image-root data/raw/plantvillage --split test \
  --output research/results/raw/model_comparison/efficientnet_b0.json

python -m crop_ai.training.predict \
  --checkpoint artifacts/models/efficientnet_b0/best.pt --image path/to/leaf.jpg
```

For lesion masks, create a CSV with `image_path,mask_path,split`, then run:

```bash
python training/train_segmentation.py --config configs/training/unet.yaml
```

U-Net checkpoints remain `validated=false`. Only a held-out ground-truth mask evaluation may mark a
checkpoint validated for severity use.
