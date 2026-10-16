# Project structure

What exists where, and where new things should go.

**Rule of thumb:** Python lives in `backend/` (run Python commands there). React lives in
`frontend/` (run npm there). Documentation lives in `docs/`.

```text
crop-disease-multimodal-ai/
├── backend/            Python project root (pyproject.toml, .env, .venv)
│   ├── app/            FastAPI web API
│   ├── src/crop_ai/    Core library: CV, models, RAG, LLM, reports
│   ├── training/       Thin CLI entry points for training/evaluation
│   ├── scripts/        Dev utilities and run scripts
│   ├── tests/          pytest suites
│   ├── configs/        YAML configs for data, training, RAG, experiments
│   ├── data/           Datasets and vector indexes        (contents git-ignored)
│   ├── artifacts/      Model checkpoints, runtime DBs      (contents git-ignored)
│   ├── knowledge/      RAG source documents + manifest     (documents git-ignored)
│   └── research/       Protocols, results, plots, tables   (outputs git-ignored)
├── frontend/           React + TypeScript + Vite app
├── docs/               Guides and architecture notes
├── README.md           Quick start
├── AGENTS.md           Rules, folder map and commands for any AI assistant (give this file)
└── CLAUDE.md           Imports AGENTS.md for Claude Code
```

## Backend: `backend/app/` (web API)

Only HTTP concerns go here: requests, validation, wiring. Domain logic belongs in `crop_ai`.

| Path | Contains | Put new… here |
|---|---|---|
| `app/main.py` | App creation, CORS, router registration | Register a new router |
| `app/core/config.py` | All settings (read from `backend/.env`) | New setting: add it here **and** to `.env.example` |
| `app/api/routes/` | Endpoints: `diagnosis`, `chat`, `knowledge`, `reports`, `realtime` (WebSocket), `system`, `health`, `feedback`, `research` | New endpoint: a route file, or extend the matching one |
| `app/schemas/` | Pydantic request/response models (`chat`, `diagnosis`, `system`) | API request/response types |
| `app/services/` | Orchestration: `diagnosis_service`, `chat_service`, `llm_service`, `knowledge_service`, `model_manager`, `session_store`, `diagnosis_archive` | Logic that combines several `crop_ai` pieces for a request |

## Backend: `backend/src/crop_ai/` (core library)

Framework-independent. It must not import FastAPI or `app`.

| Package | Contains | Put new… here |
|---|---|---|
| `cv/` | Image decode, preprocessing, leaf/lesion segmentation, heuristic severity | Image processing steps |
| `models/` | Classifier factory (ResNet50/EfficientNet-B0/MobileNetV2), mock classifier, checkpoints, registry | New model architecture |
| `training/` | Dataset, augmentations, losses, trainer, train/evaluate/predict | Training-loop changes |
| `segmentation/` | U-Net, losses, metrics, training, inference | Segmentation models |
| `explainability/` | Grad-CAM | Other explainability methods |
| `evidence/` | Builds the typed diagnosis evidence passed to the LLM | New evidence fields |
| `data/` | Dataset manifests, splits, overlap/leakage checks | Dataset preparation |
| `evaluation/` | Metrics and plots | New metrics |
| `rag/` | Ingest, chunking, TF-IDF/semantic/hybrid retrievers, index build, RAG eval | Retrieval changes |
| `llm/` | Provider interface (`base.py`), mock/OpenAI/Gemini/Ollama providers, answer formatting, prompts, citation grounding | New LLM provider: a `*_provider.py` file plus a line in `factory.py`; agent/tool logic |
| `reports/` | HTML/PDF/JSON diagnosis reports | Report formats |
| `experiments/` | Experiment runner, RAG grounding experiment | New research experiments |
| `device.py` | CUDA → MPS → CPU selection | Never hard-code `.cuda()`; use this |

## Backend: everything else

| Path | Contains | Put new… here |
|---|---|---|
| `training/` | CLI entry points that call `crop_ai.training` / `segmentation` | Only thin wrappers; logic goes in `crop_ai` |
| `scripts/` | `run_backend.sh`, `run_frontend.sh`, `build_knowledge_base.sh`, `fetch_knowledge_sources.py`, `check_llm.py`, `check_environment.py`, `check_device.py`, smoke tests, synthetic data generator | Dev/ops helper scripts |
| `tests/unit/` | Fast tests per module | Tests for `crop_ai` and services |
| `tests/integration/` | API tests (`test_api.py`) | Endpoint tests |
| `tests/ml/` | Tests needing torch (skipped without the `ml` extra) | Model/segmentation tests |
| `tests/fixtures/` | Small synthetic test data | Test fixtures (never real datasets) |
| `configs/data/` | `plantvillage.yaml`, `plantdoc.yaml` | New dataset config |
| `configs/training/` | `resnet50`, `efficientnet_b0`, `mobilenet_v2`, `unet` | New training run config |
| `configs/*.yaml` | `inference.yaml`, `rag.yaml`, `experiments.yaml` | |
| `data/raw/` | Original datasets (e.g. `data/raw/plantvillage/`) | Downloaded datasets, **never committed** |
| `data/processed/` | Preprocessed data | Generated |
| `data/manifests/` | Created by `crop_ai.data.prepare` (not git-ignored) | Generated split CSVs |
| `data/vectorstore/` | TF-IDF / FAISS indexes, `chunks.json` | Generated by the RAG index build |
| `artifacts/models/` | Trained checkpoints (`model_state.pt`, `labels.json`, `metadata.json`, `preprocessing.json`) | Trained models, **never committed** |
| `artifacts/runtime/` | `chat.sqlite3`, `feedback.sqlite3`, saved diagnoses | Created at runtime |
| `knowledge/source_docs/` | RAG source PDFs/text (~207 extension publications from `scripts/fetch_knowledge_sources.py`) | Agricultural documents with a known licence |
| `knowledge/source_manifest.csv` | One row per source document (written by the fetch script; hand-added rows are kept) | Register every document here |
| `research/protocols/` | RAG evaluation protocol, sample benchmarks | Experiment protocols and benchmarks |
| `research/results/`, `plots/`, `tables/` | Experiment outputs | Generated; only real, reproducible results |
| `.env.example` | Template for all settings | Every new setting, with a safe default |
| `.env` | Your local settings and API keys | **Never committed** |

## Frontend: `frontend/`

| Path | Contains | Put new… here |
|---|---|---|
| `src/main.tsx`, `src/App.tsx` | Entry point and routes | Register a new page route |
| `src/pages/` | `HomePage`, `DiagnosePage`, `AskCropAIPage`, `LiveCameraPage`, `ResearchPage`, `AboutPage` | New full page |
| `src/components/` | `ChatPanel`, `DiagnosisResults`, `DemoBanner`, `Navbar` | Reusable UI pieces |
| `src/api/client.ts` | All backend calls (REST and WebSocket URLs) | New API call; don't call `fetch` from components |
| `src/types/api.ts` | TypeScript types mirroring backend schemas | Keep in sync with `backend/app/schemas/` |
| `src/styles/main.css` | Global styles | Styles |
| `vite.config.ts` | Dev server (port 5173, fixed) | Build/dev config |
| `.nvmrc`, `package.json` `engines` | Node version (22 / ≥20.19) | |

## Docs: `docs/`

Guides: `architecture.md`, `api.md`, `rag_guide.md`, `knowledge_base_guide.md`, `llm_setup.md`,
`openai_setup.md`, `ollama_setup.md`, `chatbot_architecture.md`, `training_guide.md`,
`dataset_guide.md`, `model_architecture.md`, `gpu_handoff.md`, `report_generation.md`,
`research_protocol.md`, `research_experiments.md`, `llm_research_protocol.md`, `demo_guide.md`,
`presentation_demo.md`. Progress logs: `implementation_*.md`, `phase*_progress.md`,
`CODEX_HANDOFF.md`. New guides go here. Link them from `README.md`.

## Common tasks

| I want to… | Touch these |
|---|---|
| Add an API endpoint | `app/api/routes/`, `app/schemas/`, `app/services/`, `frontend/src/api/client.ts`, `frontend/src/types/api.ts`, `tests/integration/` |
| Add an LLM provider (e.g. Claude) | `src/crop_ai/llm/<name>_provider.py`, `llm/factory.py`, `app/core/config.py`, `.env.example`, `tests/unit/` |
| Add a setting | `app/core/config.py` and `.env.example` |
| Train a model | dataset in `data/raw/`, config in `configs/training/`, output in `artifacts/models/`; see `docs/training_guide.md` |
| Fill the knowledge base | files in `knowledge/source_docs/`, rows in `source_manifest.csv`, build the index; see `docs/rag_guide.md` |
| Add a frontend page | `frontend/src/pages/`, route in `App.tsx`, link in `components/Navbar.tsx` |
