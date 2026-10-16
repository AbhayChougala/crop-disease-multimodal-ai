# Crop Disease Detection and Diagnosis Using Multimodal AI

A research-oriented prototype that combines computer vision, explainability, knowledge retrieval, and language-model guidance. The current build runs end to end in an explicitly labelled **DEMO MODE** without model weights or API credentials.

## Current implementation

- ResNet50, EfficientNet-B0, and MobileNetV2 factories, reproducible training, evaluation,
  safe state-dict checkpoints, top-k inference, calibration metadata, and abstention
- OpenCV preprocessing, U-Net lesion-segmentation training/inference, mask metrics, safe
  experimental severity, and class-specific Grad-CAM
- Manifest-only ingestion; TF-IDF, SentenceTransformers/FAISS, and hybrid retrieval; validated
  passage citations; OpenAI/Ollama provider interfaces; and deterministic offline demo chat
- FastAPI model manager, durable SQLite general/diagnosis chat, HTML/PDF/JSON reports, and
  throttled real-time WebSocket inference
- React + TypeScript diagnosis, dedicated Ask CropAI, live-camera, and research interface

No model accuracy, research result, or agricultural validation is claimed by this repository.

## Quick start

Requirements: Python 3.12+ and Node.js 20+.

Run all backend commands from the cloned repository root (`crop_disease_multimodal_ai`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ml,llm,reports]'
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
[chat architecture](docs/chatbot_architecture.md), [report guide](docs/report_generation.md),
[GPU handoff](docs/gpu_handoff.md), and [research experiments](docs/research_experiments.md).

## Phase 3 quick demo

The default offline configuration uses the mock classifier and mock LLM. Open `/ask` for general
chat or diagnose an image and choose “Open diagnosis-aware chat.” Answers remain explicitly demo and
will show no grounded agriculture claims until `RAG_ENABLED=true` and an approved corpus is indexed.
Use the report buttons to export the complete diagnosis evidence.

Build indexes and run the clearly synthetic retrieval fixture as documented in
[`docs/rag_guide.md`](docs/rag_guide.md). Semantic model download is always explicit.
