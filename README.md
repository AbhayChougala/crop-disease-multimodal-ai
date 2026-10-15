# Crop Disease Detection and Diagnosis Using Multimodal AI

A research-oriented prototype that combines computer vision, explainability, knowledge retrieval, and language-model guidance. The current build runs end to end in an explicitly labelled **DEMO MODE** without model weights or API credentials.

## Current implementation

- ResNet50, EfficientNet-B0, and MobileNetV2 factories, reproducible training, evaluation,
  safe state-dict checkpoints, top-k inference, calibration metadata, and abstention
- OpenCV preprocessing, U-Net lesion-segmentation training/inference, mask metrics, safe
  experimental severity, and class-specific Grad-CAM
- Local source-manifest ingestion, TF-IDF retrieval, passage citations, OpenAI/Ollama provider
  interfaces, and deterministic offline demo chat
- FastAPI model manager, diagnosis sessions, knowledge search, research summaries, and
  throttled real-time WebSocket inference
- React + TypeScript diagnosis/chat/live-camera/research interface

No model accuracy, research result, or agricultural validation is claimed by this repository.

## Quick start

Requirements: Python 3.12+ and Node.js 20+.

Run all backend commands from the cloned repository root (`crop_disease_multimodal_ai`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ml,llm]'
cp .env.example .env
uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is at `http://localhost:8000/docs`.

## Validation

```bash
source .venv/bin/activate
ruff check .
pytest
python scripts/ml_smoke_test.py
python scripts/training_smoke_test.py
npm --prefix frontend run build
```

Prepare data and train only after configuring local dataset paths:

```bash
python -m crop_ai.data.prepare --config configs/data/plantvillage.yaml
python -m crop_ai.training.train --config configs/training/efficientnet_b0.yaml
MODEL_MODE=real MODEL_PATH=artifacts/models/efficientnet_b0/best.pt \
  uvicorn backend.app.main:app --reload --port 8000
```

## Safety and research status

- Mock outputs are placeholders and are always returned with `is_mock: true`.
- Severity is an unvalidated color-segmentation baseline and is always labelled experimental.
- The app is decision support, not a replacement for an agricultural expert.
- Do not commit datasets, checkpoints, vector stores, credentials, or private source documents.

See the [dataset guide](docs/dataset_guide.md), [training guide](docs/training_guide.md),
[model architecture](docs/model_architecture.md), [RAG guide](docs/rag_guide.md),
[GPU handoff](docs/gpu_handoff.md), and [research experiments](docs/research_experiments.md).
