# Dataset guide

Datasets are local inputs and are intentionally excluded from Git. The classifier manifest uses
`image_path,dataset,crop,disease,class_id,split,source_id,group_id,sha256,width,height`.
Canonical class IDs use `crop__disease`. Never merge datasets until an explicit taxonomy mapping
identifies genuinely overlapping classes.

1. Place an ImageFolder-style dataset outside Git, with one canonical class folder per label.
2. Set `root`, `dataset_id`, and output paths in `configs/data/plantvillage.yaml` or
   `configs/data/plantdoc.yaml`.
3. Prepare and validate it:

```bash
python -m crop_ai.data.prepare --config configs/data/plantvillage.yaml
python -m crop_ai.data.validate --manifest data/manifests/plantvillage.csv
```

Preparation decodes each image, applies dimension/pixel limits, hashes contents, fingerprints the
manifest, and creates deterministic group-aware splits. Every class needs at least three independent
groups so train, validation, and test remain distinct. Duplicate hashes across splits are reported as
leakage and must be resolved before experiments.

PlantDoc contains field images and detection annotations in some distributions. Do not silently treat
detection data as classification data; document the conversion and label mapping used.
