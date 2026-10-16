from __future__ import annotations

import argparse
from pathlib import Path

from crop_ai.training.config import load_training_config
from crop_ai.training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a crop-disease classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    try:
        trainer = Trainer(load_training_config(arguments.config))
        history = trainer.fit(arguments.resume)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Completed {len(history)} epoch(s). Synthetic runs are not agricultural results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
