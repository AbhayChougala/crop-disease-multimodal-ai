from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    return payload


def aggregate(config_path: Path) -> dict[str, Any]:
    config = _config(config_path)
    root = Path(config["results_root"])
    studies = []
    csv_rows = []
    for definition in config["experiments"]:
        experiment_id = definition["id"]
        files = sorted((root / experiment_id).glob("*.json"))
        measurements = []
        failures = []
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload.get("metadata"), dict) or not isinstance(
                    payload.get("metrics"), dict
                ):
                    raise ValueError("missing metadata or metrics mapping")
                measurements.append({"artifact": str(path), **payload})
                for metric, value in payload["metrics"].items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        csv_rows.append(
                            {
                                "experiment_id": experiment_id,
                                "artifact": str(path),
                                "metric": metric,
                                "value": value,
                            }
                        )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append({"artifact": str(path), "error": str(exc)})
        status = (
            "complete" if measurements and not failures else "failed" if failures else "pending"
        )
        studies.append(
            {
                **definition,
                "status": status,
                "measurements": measurements,
                "failures": failures,
            }
        )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "results_available" if csv_rows else "no_results",
        "studies": studies,
    }
    summary_path = Path(config["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table_path = Path(config["table_path"])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment_id", "artifact", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate genuine experiment JSON files without inventing missing results."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    arguments = parser.parse_args()
    summary = aggregate(arguments.config)
    print(f"Experiment status: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
