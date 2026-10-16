# Crop Disease Detection and Diagnosis Using Multimodal AI

A research-oriented prototype that combines computer vision, explainability, knowledge retrieval, and language-model guidance. The current build runs end to end in an explicitly labelled **DEMO MODE** without model weights or API credentials.

## Current implementation

- ResNet50, EfficientNet-B0, and MobileNetV2 factories, reproducible training, evaluation,
  safe state-dict checkpoints, top-k inference, calibration metadata, and abstention
- OpenCV preprocessing, U-Net lesion-segmentation training/inference, mask metrics, safe
  experimental severity, and class-specific Grad-CAM
- Manifest-only ingestion; TF-IDF, SentenceTransformers/FAISS, and hybrid retrieval; validated
  passage citations; Gemini/OpenAI/Ollama providers with a scoped, structured prompt; and
  deterministic offline demo chat
- FastAPI model manager, durable SQLite general/diagnosis chat, HTML/PDF/JSON reports, and
  throttled real-time WebSocket inference
- React + TypeScript diagnosis, dedicated Ask CropAI, live-camera, and research interface

No model accuracy, research result, or agricultural validation is claimed by this repository.

## Requirements

| Tool | Supported versions | Notes |
|---|---|---|
| Python | **3.12, 3.13 or 3.14** (not 3.15+) | 3.12 is what CI uses and is recommended |
| Node.js | **20.19+ or 22.12+** | Node 22 LTS recommended (see `frontend/.nvmrc`) |
| Git | any recent | |

No GPU, model weights, dataset, or API key is needed. Without them the app runs in **DEMO MODE**.

> **Check your Python first.** `python3 --version` may point to an unsupported version (for
> example Homebrew's newest Python). Always create the virtual environment with an explicit
> version such as `python3.12`. If it is missing: macOS `brew install python@3.12`, Ubuntu
> `sudo apt install python3.12 python3.12-venv`, Windows: install 3.12 from python.org.

## Repository layout

```text
crop-disease-multimodal-ai/
├── backend/     Python: FastAPI app, ML/RAG/LLM library, training, tests, configs, data
│   ├── app/          FastAPI API
│   ├── src/crop_ai/  CV models, RAG, LLM providers, reports
│   ├── training/  scripts/  tests/  configs/
│   ├── data/  artifacts/  knowledge/  research/
│   ├── pyproject.toml  .env.example
├── frontend/    React + TypeScript + Vite
└── docs/        Documentation
```

**Rule of thumb:** every Python command runs inside `backend/`; every npm command runs inside
`frontend/`. For what each folder contains and where new code should go, see
[docs/project_structure.md](docs/project_structure.md).

## Quick start (macOS / Linux)

```bash
git clone https://github.com/AbhayChougala/crop-disease-multimodal-ai.git
```

**Terminal 1: backend**

```bash
cd crop-disease-multimodal-ai/backend
python3.12 -m venv .venv
source .venv/bin/activate
python --version                         # must print 3.12.x (or 3.13/3.14)
python -m pip install --upgrade pip
python -m pip install -e '.[dev,llm,reports]'
cp -n .env.example .env                  # optional; -n never overwrites an existing .env
python scripts/check_environment.py      # prints OK or tells you what is wrong
python -m uvicorn app.main:app --reload --reload-dir app --reload-dir src --port 8000
```

**Terminal 2: frontend**

```bash
cd crop-disease-multimodal-ai/frontend
npm ci                                   # installs exact versions from package-lock.json
npm run dev
```

Open `http://localhost:5173`. API docs are at `http://localhost:8000/docs`.

After the first setup, `backend/scripts/run_backend.sh` and `backend/scripts/run_frontend.sh` start
each side from any directory (no need to activate the venv).

## Quick start (Windows PowerShell)

```powershell
cd crop-disease-multimodal-ai\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -e ".[dev,llm,reports]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python scripts\check_environment.py
python -m uvicorn app.main:app --reload --reload-dir app --reload-dir src --port 8000
```

Second PowerShell window:

```powershell
cd crop-disease-multimodal-ai\frontend
npm ci
npm run dev
```

## Turn on the real chatbot (Gemini + knowledge base)

The steps above run everything in DEMO MODE. To get real, cited answers (still with a demo
classifier until a model is trained), do this once inside `backend/` with the venv active:

```bash
# 1. Install retrieval libraries (pulls PyTorch, ~1 GB; first build downloads a ~90 MB model)
python -m pip install -e '.[semantic-rag]'

# 2. Download ~208 extension fact sheets and build the vector store (5-15 min)
python scripts/fetch_knowledge_sources.py
scripts/build_knowledge_base.sh
```

3. Get a free Gemini API key at <https://aistudio.google.com/apikey> and set these in
   `backend/.env` (never commit this file):

   ```dotenv
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-3.8-flash
   LLM_API_KEY=<your key>
   GEMINI_BACKEND=developer
   LLM_FALLBACK_MODELS=gemini-3.7-flash,gemini-3.5-flash-lite,gemini-flash-latest
   RAG_ENABLED=true
   RAG_RETRIEVER=hybrid
   ```

4. Check it, then restart the backend:

   ```bash
   python scripts/check_llm.py          # prints "OK" and a cited answer, or the exact error
   ```

On Windows, replace `scripts/build_knowledge_base.sh` with:
`python -m crop_ai.rag.validate`, `python -m crop_ai.rag.ingest`, then
`python -m crop_ai.rag.build_index --mode all --allow-model-download`.

## Working with an AI assistant

Give the assistant **[`AGENTS.md`](AGENTS.md)**. It holds the project rules, the folder map, where
each kind of change belongs, the commands and the current state.

- **Claude Code, Codex, Cursor, Copilot and other coding agents** read `AGENTS.md` automatically
  when opened in this repository (`CLAUDE.md` just imports it).
- **ChatGPT, Claude or Gemini in a browser:** upload or paste `AGENTS.md` first, then the files you
  want changed, and say "follow AGENTS.md". For more folder detail also attach
  `docs/project_structure.md`.
- Never paste `backend/.env` or an API key into a chat.

## Optional extras

Run these inside `backend/` with the venv active.

| Extra | Install | What it enables |
|---|---|---|
| `ml` | `pip install -e '.[ml]'` | PyTorch training/inference, Grad-CAM (about 1 GB+ download) |
| `rag` | `pip install -e '.[rag]'` | TF-IDF keyword retrieval (scikit-learn, no PyTorch) |
| `semantic-rag` | `pip install -e '.[semantic-rag]'` | FAISS vector store + SentenceTransformers embeddings (pulls PyTorch) |

The default PyTorch wheel is CPU-only on Windows/Linux. For an NVIDIA GPU, install the CUDA build
from <https://pytorch.org/get-started/locally/> before `.[ml]`. Apple Silicon uses MPS
automatically. See [GPU handoff](docs/gpu_handoff.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Package ... requires a different Python: 3.15.x not in '<3.15,>=3.12'` | The venv uses an unsupported Python. Delete `.venv` and recreate it with `python3.12 -m venv .venv`. |
| `ModuleNotFoundError: No module named 'app'` / `does not appear to be a Python project` | The command was run outside `backend/`. `cd backend` first, or use `backend/scripts/run_backend.sh`. |
| `npm error enoent Could not read package.json` | npm was run outside `frontend/`. `cd frontend` first. |
| `uvicorn: command not found` | The venv is not active. Activate it, or run `python -m uvicorn ...`. |
| Frontend shows network/CORS errors | Backend not running on port 8000, or the page is not served from an origin listed in `FRONTEND_ORIGIN` (comma-separated) in `backend/.env`. |
| `Port 5173 is in use` | Another Vite server is running. Stop it; the port is fixed so it stays in the backend's CORS list. |
| `npm ci` complains about the engine / Vite fails to start | Node is too old. Install Node 22 LTS (`nvm use` in `frontend/` reads `.nvmrc`). |
| Prompt shows `(.venv)` twice | Your editor already activated the venv; harmless. Open a new terminal or run `deactivate`. |
| Chat says "insufficient evidence" with a real LLM | Expected until a knowledge corpus is indexed; see [RAG guide](docs/rag_guide.md). |
| Chat says "The configured language provider is unavailable" | Run `python scripts/check_llm.py` in `backend/`; it prints Google's exact error (bad key, retired model, rate limit). |
| Ask page stuck on "Starting session…" | The backend is restarting or not running. Start it with `scripts/run_backend.sh` (it only reloads on code changes). |

## Validation

```bash
cd backend
source .venv/bin/activate
python scripts/check_environment.py
ruff check .
pytest
python scripts/ml_smoke_test.py          # requires the ml extra
python scripts/training_smoke_test.py    # requires the ml extra
npm --prefix ../frontend run build
```

Prepare data and train only after configuring local dataset paths (from `backend/`):

```bash
python -m crop_ai.data.prepare --config configs/data/plantvillage.yaml
python -m crop_ai.training.train --config configs/training/efficientnet_b0.yaml
MODEL_MODE=real MODEL_PATH=artifacts/models/efficientnet_b0/best.pt \
  python -m uvicorn app.main:app --reload --reload-dir app --reload-dir src --port 8000
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
