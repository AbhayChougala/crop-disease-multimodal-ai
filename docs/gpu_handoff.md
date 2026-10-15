# GPU handoff

The real-model phase should run on Windows or Linux with an NVIDIA GPU.

1. Clone the repository and create a Python 3.12 virtual environment.
2. Install the PyTorch build selected for the machine's CUDA version using the official PyTorch installation selector.
3. Run `python -m scripts.check_device` and require `CUDA available: True`.
4. Install the project dependencies.
5. Place the dataset under an ignored external or `data/raw` path.
6. Run dataset inspection and a one-epoch smoke training before any full run.
7. Evaluate and export the selected state dict with `labels.json`, `metadata.json`, and `preprocessing.json`.
8. Copy the artifact directory under `artifacts/models/`, switch `MODEL_MODE=real`, and rerun the API tests.

Exact training commands will be added with the Phase 2 training implementation. Never commit datasets or model artifacts.
