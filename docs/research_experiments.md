# Research experiments

`configs/experiments.yaml` registers seven studies: model comparison, preprocessing ablation,
cross-domain robustness, explainability review, severity segmentation, LLM grounding, and runtime.
Every raw result must contain a `metadata` mapping and a `metrics` mapping. Metadata records the seed,
dataset, model, preprocessing, device, timestamp, platform, configuration snapshot, and Git commit.

Write genuine run outputs beneath `research/results/raw/<experiment-id>/`, then aggregate them:

```bash
python -m crop_ai.experiments.runner --config configs/experiments.yaml
```

The runner reports a missing study as `pending`, records malformed inputs as failures, and imports only
numeric measurements into the CSV summary. It does not fill absent results or manufacture charts.
The Research API and page display imported artifacts when present and “Pending experiment” otherwise.

Controls required before publication:

- identical saved dataset splits across classifier architectures;
- explicit preprocessing condition with all other settings held constant;
- documented overlapping taxonomy and exclusions for cross-domain tests;
- correctly classified and failure examples for Grad-CAM, with no visual-faithfulness claim;
- ground-truth lesion masks for Dice, IoU, and severity error;
- fixed questions and manual unsupported-claim/citation review for LLM grounding;
- warmup, repeated measurements, device identity, and separate stage latency for runtime results.
