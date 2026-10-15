# GPU handoff

On macOS, create Python 3.12 virtual environment and install `.[dev,ml]`; automatic device selection
is CUDA, then MPS, then CPU. Mixed precision is enabled only on CUDA.

On the NVIDIA Windows/Linux laptop:

```bash
git clone <repository-url>
cd crop_disease_multimodal_ai
python3.12 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA-compatible `torch` and `torchvision` command produced by the official PyTorch
installer for that laptop, then install the project without replacing those wheels:

```bash
python -m pip install -e '.[dev,ml]'
python scripts/check_device.py
```

Require `CUDA available: True` and record the printed GPU/PyTorch information. Place datasets under
ignored `data/raw/` paths, update the dataset YAML, and run:

```bash
python -m crop_ai.data.prepare --config configs/data/plantvillage.yaml
python -m crop_ai.data.validate --manifest data/manifests/plantvillage.csv
python -m crop_ai.training.train --config configs/training/resnet50.yaml
python -m crop_ai.training.train --config configs/training/efficientnet_b0.yaml
python -m crop_ai.training.train --config configs/training/mobilenet_v2.yaml
```

Run a one-epoch edited copy of each configuration before a long job. Resume an interrupted run with
`--resume artifacts/models/<experiment>/last.pt`. Evaluate each best checkpoint against the identical
held-out split and save its JSON beneath `research/results/raw/model_comparison/`.

Copy the selected `.pt` checkpoint back to the Mac under ignored `artifacts/models/`, preserving its
SHA-256. No source change is needed:

```bash
MODEL_MODE=real MODEL_PATH=artifacts/models/<checkpoint>.pt \
  uvicorn backend.app.main:app --reload --port 8000
curl http://localhost:8000/api/v1/system/capabilities
```

The response must report `model_mode: real` and `model_loaded: true`. Never commit datasets,
checkpoints, vector indexes, secrets, or generated result artifacts.
