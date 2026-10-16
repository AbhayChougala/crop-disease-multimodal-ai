from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment_id: str
    timestamp: str
    git_commit: str | None
    seed: int
    device: str
    python_version: str
    platform: str
    model: str
    dataset: str
    preprocessing: str
    config: dict[str, Any]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_experiment_metadata(
    experiment_id: str,
    seed: int,
    device: str,
    model: str,
    dataset: str,
    preprocessing: str,
    config: dict[str, Any],
) -> ExperimentMetadata:
    return ExperimentMetadata(
        experiment_id=experiment_id,
        timestamp=datetime.now(UTC).isoformat(),
        git_commit=git_commit(),
        seed=seed,
        device=device,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        model=model,
        dataset=dataset,
        preprocessing=preprocessing,
        config=config,
    )


def write_result(path: Path, metadata: ExperimentMetadata, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": asdict(metadata), "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
