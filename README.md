# Crop Disease Detection and Diagnosis Using Multimodal AI

A B.Tech research prototype that looks at a photo of a crop leaf, identifies the disease, measures
the visible damage, explains where the model looked, and then lets a team of AI agents give a
grounded, cited consultation: diagnosis, clarifying questions, spread risk with live weather,
treatment and a follow-up plan.

```text
Leaf photo
  → OpenCV: decode, isolate the leaf, find lesions, experimental severity %
  → EfficientNet-B0 (PyTorch, 38 PlantVillage classes): disease, confidence, top-5
  → Grad-CAM heatmap (where the model looked)
  → Evidence package (prediction + severity + uncertainty flags)
       ├→ Diagnose page: results, images, short advice, PDF/HTML report
       ├→ Diagnosis chat: a Gemini orchestrator runs the agent team → one reply
       └→ Agents page: run any single agent yourself
            ↳ agents search a knowledge base of ~207 extension fact sheets (hybrid TF-IDF + FAISS)
            ↳ the risk agent reads a free 3-day weather forecast (Open-Meteo)
            ↳ a safety check removes invented citations and uncited chemical advice
```

The system is decision support, not a replacement for an agricultural expert. No accuracy figure
or research result is claimed by this repository until it is measured on held-out data.

## What works today

| Part | Status |
|---|---|
| Disease classifier | EfficientNet-B0 trained on PlantVillage (38 crop/disease classes), downloaded with one script, runs on CUDA, Apple MPS or CPU |
| Image analysis | Leaf isolation, lesion overlay, severity % (always labelled experimental), Grad-CAM |
| Knowledge base | ~207 extension publications (UC IPM, TNAU, NC State, Ohio State, UF, WSU, ...) in 1,292 chunks |
| Chatbot | Gemini (or OpenAI / Ollama) with retrieved, cited passages; short structured answers; refuses off-topic questions |
| Agent team | LLM orchestrator + evidence, diagnosis, question, risk/weather, treatment, follow-up/reassess and Q&A agents, with step/call budgets and a rule-based fallback ([details](docs/agent_architecture.md)) |
| Agents page | Run each agent on its own; results are remembered per photo so agents build on each other |
| Frontend | Diagnose, Live Camera, Ask CropAI (chat), Agents, Research pages |
| Reports | HTML / PDF / JSON diagnosis reports |
| Demo mode | Everything also runs with no model, no API key and no internet, clearly labelled **DEMO MODE** |

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | **3.12, 3.13 or 3.14** (not 3.15+) | 3.12 is what CI uses and is recommended |
| Node.js | **20.19+ or 22.12+** | Node 22 LTS recommended (`frontend/.nvmrc`) |
| Git | any recent | |
| Disk | ~3 GB | PyTorch, the embedding model, the knowledge base |
| Gemini API key | optional, free | <https://aistudio.google.com/apikey>; without it the chat runs in DEMO MODE |

A GPU is optional. Apple Silicon uses MPS automatically; NVIDIA needs the CUDA build of PyTorch
(see [Optional: GPU](#optional-nvidia-gpu)).

> **Check your Python first.** `python3 --version` may point to an unsupported version (for
> example Homebrew's newest Python). Always create the virtual environment with an explicit
> version such as `python3.12`. If it is missing: macOS `brew install python@3.12`, Ubuntu
> `sudo apt install python3.12 python3.12-venv`, Windows: install 3.12 from python.org.

## Repository layout

```text
crop-disease-multimodal-ai/
├── backend/            Python project root. Run ALL Python commands here.
│   ├── app/            FastAPI web API (routes, schemas, services)
│   ├── src/crop_ai/    Library: cv/, models/, rag/, llm/, agents/, reports/, training/, ...
│   ├── model/          download_model.py; the checkpoint goes to model/checkpoints/ (git-ignored)
│   ├── scripts/        run_backend.sh, check_environment.py, check_llm.py, knowledge-base scripts
│   ├── tests/          unit/, integration/, ml/
│   ├── knowledge/      source_manifest.csv (+ downloaded fact sheets, git-ignored)
│   ├── data/ artifacts/ research/   generated content (git-ignored)
│   ├── pyproject.toml  requirements.txt  .env.example
├── frontend/           React + TypeScript + Vite. Run ALL npm commands here.
├── docs/               Guides and architecture notes
└── AGENTS.md           Rules and map for AI coding assistants (CLAUDE.md imports it)
```

Full per-folder table and where new code should go: [docs/project_structure.md](docs/project_structure.md).

## Full setup (macOS / Linux)

Do this once. Steps 4–6 are what turn DEMO MODE into the real system; you can stop after step 3
for a quick demo-only run.

### 1. Clone

```bash
git clone https://github.com/AbhayChougala/crop-disease-multimodal-ai.git
cd crop-disease-multimodal-ai/backend
```

### 2. Create the Python environment and install everything

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version                         # must print 3.12.x (or 3.13/3.14)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # everything, ~2 GB (mostly PyTorch)
```

`requirements.txt` just selects every optional group from `pyproject.toml`. For a smaller,
demo-only install use `python -m pip install -e '.[dev,llm,reports]'` instead (see
[Dependency groups](#dependency-groups)).

### 3. Create your settings file

```bash
cp -n .env.example .env                  # -n never overwrites an existing .env
python scripts/check_environment.py      # prints OK or tells you what is wrong
```

`backend/.env` holds your settings and API key. It is git-ignored: never commit it or paste it
into a chat. At this point the app works in DEMO MODE.

### 4. Download the disease model

```bash
python model/download_model.py           # 16 MB, SHA-256 verified, saved to model/checkpoints/
```

Then set in `backend/.env`:

```dotenv
MODEL_MODE=real
MODEL_PATH=model/checkpoints/best_model.pth
```

### 5. Build the knowledge base

```bash
python scripts/fetch_knowledge_sources.py   # downloads ~208 extension fact sheets
scripts/build_knowledge_base.sh             # builds data/vectorstore/ (5-15 min; first run
                                            # downloads a ~90 MB embedding model)
```

Then set in `backend/.env`:

```dotenv
RAG_ENABLED=true
RAG_RETRIEVER=hybrid
```

### 6. Connect Gemini

Get a free key at <https://aistudio.google.com/apikey> and set in `backend/.env`:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
LLM_API_KEY=<your key>
GEMINI_BACKEND=developer
LLM_FALLBACK_MODELS=gemini-3.8-flash,gemini-3.7-flash,gemini-flash-latest
```

Check it:

```bash
python scripts/check_llm.py              # prints "OK" and a cited answer, or the exact error
```

Google retires model names from time to time; if a model returns 404, pick a current one in AI
Studio. The fallback list is tried automatically on 404 / 429 / 503.

### 7. Install the frontend

```bash
cd ../frontend
npm ci                                   # installs exact versions from package-lock.json
```

### 8. Run it

**Terminal 1 – backend**

```bash
cd crop-disease-multimodal-ai/backend
scripts/run_backend.sh                   # uses .venv automatically; API on http://localhost:8000
```

**Terminal 2 – frontend**

```bash
cd crop-disease-multimodal-ai/frontend
npm run dev                              # http://localhost:5173
```

Open **http://localhost:5173**. API docs: <http://localhost:8000/docs>.

### 9. Confirm everything is live

| Check | Expected |
|---|---|
| <http://localhost:8000/api/v1/models> | `"state": "REAL"` (classifier loaded) |
| <http://localhost:8000/api/v1/llm/status> | `"provider": "gemini"`, `"state": "CONFIGURED"` |
| Diagnose page, upload a leaf | No DEMO banner; disease, confidence, Grad-CAM |
| Agents page, run **Doctor** | Steps show "done", sources are listed |

## Setup on Windows (PowerShell)

```powershell
cd crop-disease-multimodal-ai\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python scripts\check_environment.py
python model\download_model.py
python scripts\fetch_knowledge_sources.py
python -m crop_ai.rag.validate
python -m crop_ai.rag.ingest
python -m crop_ai.rag.build_index --mode all --allow-model-download
python scripts\check_llm.py
python -m uvicorn app.main:app --reload --reload-dir app --reload-dir src --port 8000
```

Edit `backend\.env` as in steps 4–6 above. In a second PowerShell window:

```powershell
cd crop-disease-multimodal-ai\frontend
npm ci
npm run dev
```

## Running it again later

```bash
backend/scripts/run_backend.sh           # terminal 1 (from the repo root or anywhere)
backend/scripts/run_frontend.sh          # terminal 2
```

No need to activate the venv for these scripts. The backend only reloads on code changes in
`app/` and `src/`, never on data or index writes.

## Using the app

| Page | What to do |
|---|---|
| **Diagnose** | Upload a leaf photo. You get the disease, confidence, top-5, severity, leaf/lesion images, Grad-CAM, a short advisory and report export. "Open diagnosis-aware chat" continues in the chat. |
| **Live Camera** | Point the camera at a leaf for throttled real-time predictions. |
| **Ask CropAI** | General crop-health chat, or a diagnosis chat answered by the agent team. Replies show clarifying-question chips, a risk badge, the re-check date and a "How CropAI worked this out" trace. **Share location** adds weather; **Reassess with new photo** compares a new photo with the old one. |
| **Agents** | Analyse a photo once, then run a single agent: Doctor, Evidence researcher, Question agent, Risk & weather, Treatment, Follow-up planner or Q&A. Missing prerequisites run first automatically; "Team memory" shows what is already known; the raw JSON output is available for the research report. |
| **Research** | Runtime status and research tooling. |

## Demo mode vs real mode

Each piece switches independently; anything not configured stays clearly labelled.

| Piece | Real when | Otherwise |
|---|---|---|
| Classifier | `MODEL_MODE=real`, checkpoint downloaded, `ml` group installed | `MODEL_MODE=mock`: DEMO placeholder predictions (`is_mock: true`) |
| Knowledge base | `RAG_ENABLED=true` and the index is built | No citations; real LLMs are never called without evidence |
| Language model | `LLM_PROVIDER=gemini` (or openai/ollama) with a key | Mock provider: deterministic demo text; agents use rule-based fallbacks |
| Weather | `WEATHER_ENABLED=true` and the user shares a location | Risk is computed without weather |

If `MODEL_MODE=real` but the checkpoint or PyTorch is missing, the model state is `UNAVAILABLE`
and diagnosis returns an error rather than silently faking a result.

## Important settings (`backend/.env`)

All settings with defaults are in [`backend/.env.example`](backend/.env.example).

| Setting | Meaning |
|---|---|
| `MODEL_MODE`, `MODEL_PATH` | `mock` or `real`, and the checkpoint path |
| `CONFIDENCE_THRESHOLD` | Below this the classifier abstains / the case is treated as uncertain (default 0.65) |
| `RAG_ENABLED`, `RAG_RETRIEVER` | Knowledge-base search on/off; `tfidf`, `semantic` or `hybrid` |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` | Language model; `mock`, `gemini`, `openai` or `ollama` |
| `LLM_FALLBACK_MODELS` | Gemini models tried when the main one is retired, overloaded or rate limited |
| `AGENTS_ENABLED` | Use the agent team for diagnosis chat (else a single LLM call) |
| `AGENT_MAX_STEPS`, `AGENT_MAX_LLM_CALLS` | Per-turn budgets (7 steps, 10 calls). Lower `AGENT_MAX_LLM_CALLS` for faster demos |
| `AGENT_ORCHESTRATOR_MODEL` | Optional cheaper model just for the orchestrator's routing decisions |
| `WEATHER_ENABLED` | Free Open-Meteo forecast for the risk agent |
| `FRONTEND_ORIGIN` | Comma-separated browser origins allowed by CORS |

## Main API endpoints

All under `http://localhost:8000/api/v1` (interactive docs at `/docs`).

| Endpoint | Purpose |
|---|---|
| `POST /diagnose` | Upload a photo; returns prediction, severity, images, evidence and a `session_id` |
| `GET /models` | Classifier runtime state (`REAL` / `DEMO` / `UNAVAILABLE`) |
| `POST /chat/sessions` | Start a general chat, or a diagnosis chat with `diagnosis_id` |
| `POST /chat/sessions/{id}/messages` | Send a message (optional `location`); diagnosis chats use the agent team |
| `POST /chat/sessions/{id}/reassess` | Link a new photo's diagnosis and compare it with the old one |
| `POST /agents/{agent}/run` | Run one agent (`diagnose`, `retrieve_evidence`, `ask_farmer`, `assess_risk`, `recommend_treatment`, `plan_followup`, `answer_question`) |
| `GET` / `DELETE /agents/memory/{session_id}` | See or clear what the agents remember for a diagnosis |
| `GET /llm/status` | Which language model is configured |
| `WS /realtime/ws` | Live-camera inference |

## About the disease model

- **Architecture:** EfficientNet-B0, 224×224 input, ImageNet normalisation, 38 PlantVillage
  classes (apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry,
  soybean, squash, strawberry, tomato; diseases and healthy).
- **Source:** trained outside this repository and published as a GitHub release;
  `model/download_model.py` downloads it and checks its SHA-256. The legacy checkpoint format is
  converted on load in `src/crop_ai/models/checkpoints.py`.
- **Accuracy:** the validation score stored in the checkpoint comes from the external training
  run, not from this project. Measure accuracy on held-out images before reporting any figure.
  PlantVillage photos are lab-style (one leaf, plain background); expect lower accuracy on field
  photos.
- **Train your own:** prepare a dataset and run training (from `backend/`):

  ```bash
  python -m crop_ai.data.prepare --config configs/data/plantvillage.yaml
  python -m crop_ai.training.train --config configs/training/efficientnet_b0.yaml
  ```

  Then point `MODEL_PATH` at the new checkpoint. See the [training guide](docs/training_guide.md)
  and [dataset guide](docs/dataset_guide.md).

## Dependency groups

There is one `backend/requirements.txt` (installs everything). To pick groups yourself, run these
inside `backend/` with the venv active:

| Group | Install | What it enables |
|---|---|---|
| (core) | `pip install -e .` | FastAPI, OpenCV, NumPy: DEMO mode |
| `dev` | `pip install -e '.[dev]'` | pytest, ruff |
| `llm` | `pip install -e '.[llm]'` | Gemini, OpenAI, PDF reading |
| `reports` | `pip install -e '.[reports]'` | PDF reports |
| `ml` | `pip install -e '.[ml]'` | PyTorch + torchvision: real classifier, Grad-CAM, training (~1 GB) |
| `rag` | `pip install -e '.[rag]'` | TF-IDF keyword retrieval only (no PyTorch) |
| `semantic-rag` | `pip install -e '.[semantic-rag]'` | FAISS + sentence-transformers vector search (pulls PyTorch) |

Groups combine: `pip install -e '.[dev,llm,reports,ml]'`. Frontend dependencies are in
`frontend/package.json` and install with `npm ci`.

### Optional: NVIDIA GPU

The default PyTorch wheel is CPU-only on Windows/Linux. Install the CUDA build from
<https://pytorch.org/get-started/locally/> **before** `requirements.txt` / `.[ml]`. Apple Silicon
uses MPS automatically. See [GPU handoff](docs/gpu_handoff.md).

## Tests and checks

```bash
cd backend
source .venv/bin/activate
python scripts/check_environment.py
ruff check .
pytest                                   # tests force mock mode and never call paid APIs
python scripts/ml_smoke_test.py          # requires the ml group
npm --prefix ../frontend run build
```

GitHub Actions runs lint, tests and the frontend build on every push
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Working with an AI assistant

Give the assistant **[`AGENTS.md`](AGENTS.md)**. It holds the project rules, the folder map, where
each kind of change belongs, the commands and the current state.

- **Claude Code, Codex, Cursor, Copilot and other coding agents** read `AGENTS.md` automatically
  when opened in this repository (`CLAUDE.md` just imports it).
- **ChatGPT, Claude or Gemini in a browser:** upload or paste `AGENTS.md` first, then the files you
  want changed, and say "follow AGENTS.md". For more folder detail also attach
  `docs/project_structure.md`.
- Never paste `backend/.env` or an API key into a chat.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Package ... requires a different Python: 3.15.x not in '<3.15,>=3.12'` | The venv uses an unsupported Python. Delete `.venv` and recreate it with `python3.12 -m venv .venv`. |
| `ModuleNotFoundError: No module named 'app'` / `does not appear to be a Python project` | The command was run outside `backend/`. `cd backend` first, or use `backend/scripts/run_backend.sh`. |
| `npm error enoent Could not read package.json` | npm was run outside `frontend/`. `cd frontend` first. |
| `uvicorn: command not found` | The venv is not active. Activate it, or run `python -m uvicorn ...`. |
| `/models` shows `"state": "UNAVAILABLE"` | The checkpoint or PyTorch is missing. Run `python model/download_model.py` and install the `ml` group, check `MODEL_PATH`, then restart. The `error` field says exactly what failed. |
| `No module named 'torchvision'` | Install the `ml` group: `python -m pip install -e '.[ml]'`. |
| pip stops with `ResolutionImpossible` or hangs | Usually a network interruption. Upgrade pip and retry; on very slow connections install PyTorch first (`pip install torch torchvision`) then `requirements.txt`. |
| Frontend shows network/CORS errors | Backend not running on port 8000, or the page is not served from an origin listed in `FRONTEND_ORIGIN` in `backend/.env`. |
| `Port 5173 is in use` | Another Vite server is running. Stop it; the port is fixed so it stays in the backend's CORS list. |
| `npm ci` complains about the engine / Vite fails to start | Node is too old. Install Node 22 LTS (`nvm use` in `frontend/` reads `.nvmrc`). |
| Chat says "insufficient evidence" with a real LLM | The knowledge base is not built or `RAG_ENABLED=false`; see step 5. |
| Chat says "The configured language provider is unavailable" | Run `python scripts/check_llm.py`; it prints Google's exact error (bad key, retired model, rate limit). |
| Agent replies take 30–60 s | Gemini free-tier latency or rate limits. Lower `AGENT_MAX_LLM_CALLS` (e.g. 5) or set a faster `AGENT_ORCHESTRATOR_MODEL`. Steps that hit limits fall back to rules and are marked "rule-based". |
| Ask page stuck on "Starting session…" | The backend is restarting or not running. Start it with `scripts/run_backend.sh` (it only reloads on code changes). |
| Prompt shows `(.venv)` twice | Your editor already activated the venv; harmless. |

## Safety and research status

- Mock outputs are placeholders and are always returned with `is_mock: true` and a visible DEMO
  MODE label.
- Severity is an unvalidated colour-segmentation baseline and is always labelled experimental.
- Grad-CAM shows where the classifier looked; it is not a disease segmentation mask.
- The spread-risk score is an unvalidated rule-based heuristic.
- Chemical advice is only shown when it cites a retrieved fact sheet; always follow the product
  label and local regulations.
- Do not commit datasets, checkpoints, vector stores, credentials, or downloaded documents.

## Further documentation

[Agent architecture](docs/agent_architecture.md) ·
[Project structure](docs/project_structure.md) ·
[Architecture](docs/architecture.md) ·
[Model architecture](docs/model_architecture.md) ·
[Training guide](docs/training_guide.md) ·
[Dataset guide](docs/dataset_guide.md) ·
[RAG guide](docs/rag_guide.md) ·
[Knowledge base guide](docs/knowledge_base_guide.md) ·
[LLM setup](docs/llm_setup.md) ·
[Chat architecture](docs/chatbot_architecture.md) ·
[Report generation](docs/report_generation.md) ·
[GPU handoff](docs/gpu_handoff.md) ·
[Research protocol](docs/research_protocol.md) ·
[Research experiments](docs/research_experiments.md)
