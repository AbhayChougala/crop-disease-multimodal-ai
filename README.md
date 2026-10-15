# Crop Disease Detection and Diagnosis Using Multimodal AI

A research-oriented prototype that combines computer vision, explainability, knowledge retrieval, and language-model guidance. The current build runs end to end in an explicitly labelled **DEMO MODE** without model weights or API credentials.

## Current implementation

- FastAPI backend with health, capabilities, preprocessing, diagnosis, chat, feedback, research-summary, and realtime WebSocket endpoints
- Modular OpenCV preprocessing, heuristic leaf/lesion segmentation, and experimental affected-area calculation
- Deterministic mock classifier and mock advisory provider
- React + TypeScript interface for uploads and live-camera inference
- Core unit and integration tests

No model accuracy, research result, or agricultural validation is claimed by this repository.

## Quick start

Requirements: Python 3.12+ and Node.js 20+.

Run all backend commands from the cloned repository root (`crop_disease_multimodal_ai`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
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
cd frontend && npm run build
```

## Safety and research status

- Mock outputs are placeholders and are always returned with `is_mock: true`.
- Severity is an unvalidated color-segmentation baseline and is always labelled experimental.
- The app is decision support, not a replacement for an agricultural expert.
- Do not commit datasets, checkpoints, vector stores, credentials, or private source documents.

See [architecture](docs/architecture.md), [implementation plan](docs/implementation_plan.md), [implementation status](docs/implementation_status.md), and [research protocol](docs/research_protocol.md).
