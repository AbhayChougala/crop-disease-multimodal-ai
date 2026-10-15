from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crop_ai.device import get_device
from crop_ai.evaluation.metrics import classification_metrics
from crop_ai.evaluation.plots import save_confusion_matrix
from crop_ai.experiments.metadata import build_experiment_metadata, write_result
from crop_ai.models.checkpoints import load_model_from_checkpoint
from crop_ai.models.factory import build_inference_transform, parameter_count
from crop_ai.training.dataset import ManifestDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("research/results/evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    device = get_device()
    if not isinstance(device, torch.device):
        device = torch.device("cpu")
    model, labels, metadata, transform_config = load_model_from_checkpoint(
        arguments.checkpoint, device
    )
    mapping = {label.class_id: index for index, label in labels.items()}
    dataset = ManifestDataset(
        arguments.manifest,
        arguments.image_root,
        arguments.split,
        build_inference_transform(transform_config),
        mapping,
    )
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    latencies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(device)
            started = time.perf_counter()
            logits = model(images)
            latencies.append(1000 * (time.perf_counter() - started) / images.size(0))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch_targets.numpy())
    values = np.concatenate(probabilities)
    actual = np.concatenate(targets)
    class_names = [labels[index].class_id for index in range(len(labels))]
    metrics = classification_metrics(actual, values, class_names)
    metrics.update(
        {
            "mean_inference_latency_ms": float(np.mean(latencies)),
            "parameter_count": parameter_count(model),
            "model_size_bytes": arguments.checkpoint.stat().st_size,
        }
    )
    experiment = build_experiment_metadata(
        experiment_id=f"evaluation-{metadata.architecture}-{arguments.split}",
        seed=metadata.seed,
        device=str(device),
        model=metadata.architecture,
        dataset=metadata.training_dataset or "unspecified",
        preprocessing="checkpoint_transform",
        config={
            "checkpoint": str(arguments.checkpoint),
            "manifest": str(arguments.manifest),
            "split": arguments.split,
            "dataset_fingerprint": metadata.dataset_fingerprint,
        },
    )
    write_result(arguments.output, experiment, metrics)
    save_confusion_matrix(
        metrics["normalized_confusion_matrix"],
        class_names,
        arguments.output.with_suffix(".confusion_matrix.png"),
        "Normalized confusion matrix",
    )
    print(f"Wrote actual computed metrics to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
