# Crop Disease Detection and Diagnosis Using Multimodal AI

Instructions for AI coding assistants (Claude, ChatGPT/Codex, Cursor, Copilot, Gemini) and humans.
This repository is an academic B.Tech AI/ML research project. **Read this whole file before changing
anything.** If you are a chat assistant without repository access, ask the user to paste the files
you need instead of guessing their contents.

## Core rules

- Never fabricate model metrics, experiment results, citations, predictions, or research findings.
- Any mock inference must be explicitly marked `is_mock=true` and the UI must visibly show DEMO MODE.
- Never commit secrets, `.env`, datasets, model checkpoints, vector stores, or private source documents.
- Keep all ML code device-agnostic: CUDA > MPS > CPU (use `crop_ai/device.py`).
- Never hard-code `.cuda()`.
- Computer-vision preprocessing used for inference must match the selected training configuration.
- Grad-CAM is explainability, not a disease segmentation ground-truth mask.
- Severity produced from heuristic lesion segmentation must be labelled experimental unless validated against annotated masks.
- LLM agricultural statements must be grounded in retrieved sources where possible.
- The system is decision support, not a replacement for an agricultural expert.
- Keep frontend, backend, training, evaluation, and research code reproducible.
- Run relevant tests, lint, and build checks after changes.
- Do not perform long model training or download large datasets unless explicitly requested.
- Prefer maintainable implementations suitable for a student research project over unnecessary complexity.
- Use current official provider documentation (OpenAI, Google Gemini) when changing LLM integration.

## Where things go

```text
crop-disease-multimodal-ai/
├── backend/            Python project root: pyproject.toml, .env, .venv. Run ALL Python here.
│   ├── app/            FastAPI web layer only (import name `app`, not `backend.app`)
│   │   ├── main.py         app, CORS, router registration
│   │   ├── core/config.py  every setting (read from backend/.env)
│   │   ├── api/routes/     HTTP endpoints
│   │   ├── schemas/        Pydantic request/response models
│   │   └── services/       orchestration that combines crop_ai pieces per request
│   ├── src/crop_ai/    Framework-independent library. Must NOT import FastAPI or `app`.
│   │   ├── cv/ models/ training/ segmentation/ explainability/ evidence/ data/ evaluation/
│   │   ├── rag/            ingest, chunking, TF-IDF / FAISS / hybrid retrieval
│   │   ├── llm/            providers (mock, openai, gemini, ollama), prompts.py, formatting.py
│   │   └── reports/ experiments/ device.py
│   ├── training/       thin CLI wrappers only; logic belongs in crop_ai
│   ├── scripts/        run_backend.sh, run_frontend.sh, check_environment.py, check_llm.py,
│   │                   fetch_knowledge_sources.py, build_knowledge_base.sh, smoke tests
│   ├── tests/          unit/, integration/ (API), ml/ (needs torch), fixtures/, conftest.py
│   ├── configs/        YAML for data, training, RAG, experiments
│   ├── knowledge/      source_manifest.csv (committed) + source_docs/ (git-ignored)
│   ├── data/ artifacts/ research/   generated content, git-ignored
├── frontend/           React + TypeScript + Vite. Run ALL npm commands here.
│   └── src/            pages/, components/, api/client.ts (all backend calls), types/api.ts
├── docs/               guides; docs/project_structure.md has the full per-folder table
├── AGENTS.md           this file
└── CLAUDE.md           imports this file for Claude Code
```

Placement rules:

| Task | Files to touch |
|---|---|
| New API endpoint | `app/api/routes/`, `app/schemas/`, `app/services/`, `frontend/src/api/client.ts`, `frontend/src/types/api.ts`, `tests/integration/` |
| New LLM provider | `src/crop_ai/llm/<name>_provider.py` (return `advisory_payload(...)`), `llm/factory.py`, `app/core/config.py`, `.env.example`, `tests/unit/` |
| Change chatbot behaviour or tone | `src/crop_ai/llm/prompts.py` (ROLE/SCOPE/SAFETY/STYLE), `llm/schemas.py`, `llm/formatting.py` |
| New setting | `app/core/config.py` **and** `backend/.env.example` |
| New knowledge source | add to `SOURCES` in `scripts/fetch_knowledge_sources.py`, then run it and `scripts/build_knowledge_base.sh` |
| New frontend page | `frontend/src/pages/`, route in `App.tsx`, link in `components/Navbar.tsx` |
| Model/training code | `src/crop_ai/models/` or `training/`, config in `configs/training/`, tests in `tests/ml/` |

Never put Python files at the repository root or in `frontend/`, never put frontend code in
`backend/`, and never put datasets, checkpoints, API keys or downloaded documents anywhere tracked
by git.

## Commands

```bash
# backend (from backend/), Python 3.12-3.14 only
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,llm,reports]'        # + rag / semantic-rag / ml when needed
python scripts/check_environment.py
scripts/run_backend.sh                                # uvicorn on :8000, reloads on app/ src/ only
ruff check . && pytest                                # must pass before finishing

# frontend (from frontend/), Node 20.19+ or 22.12+
npm ci && npm run dev                                 # :5173
npm run build                                         # must pass before finishing

# knowledge base (from backend/, needs '.[semantic-rag]')
python scripts/fetch_knowledge_sources.py             # downloads ~208 extension publications
scripts/build_knowledge_base.sh                       # builds data/vectorstore/
python scripts/check_llm.py                           # one real LLM call through RAG
```

## Current state

- Classifier: no trained checkpoint yet, so `MODEL_MODE=mock` and diagnoses are DEMO placeholders.
- Chatbot: retrieve passages, then the LLM returns a structured `GroundedAdvisory`, citation IDs are
  intersected with retrieved IDs, and `format_advisory` renders the text. The LLM explains the
  classifier's result; it never classifies. Real providers are never called when retrieval is empty.
- Scope: the prompt only allows crop/plant-health topics; off-topic answers set `in_scope=false`
  and `advisory_payload` strips all advice and citations (status `out_of_scope`).
- LLM: Gemini via `google-genai` (`LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-3.8-flash`,
  `GEMINI_BACKEND=developer`, `LLM_FALLBACK_MODELS` for 404/429/503). OpenAI, Ollama and mock also
  exist. Google retires model names; check AI Studio if a model returns 404.
- RAG: ~207 extension publications (UC IPM, TNAU India, NC State, Ohio State, UF, WSU, ...) indexed
  into 1,292 chunks, hybrid TF-IDF + FAISS (all-MiniLM-L6-v2). Hybrid scores are min-max
  normalised, so `RAG_MIN_SCORE` cannot filter off-topic queries.

## Gotchas

- CORS origins come from comma-separated `FRONTEND_ORIGIN` in `backend/.env`; Vite port 5173 is fixed.
- `tests/conftest.py` forces mock/offline settings, so tests never read `.env` or call paid APIs.
- Run uvicorn with `--reload-dir app --reload-dir src`; plain `--reload` restarts on every data or
  index write under `backend/` and the UI hangs on "Starting session…".
- Provider `stream()` is not real token streaming; chat uses REST.
- Do not edit a user's `backend/.env` without asking; it holds their API key.

## Further docs

`docs/project_structure.md` (full folder table), `docs/architecture.md`, `docs/rag_guide.md`,
`docs/llm_setup.md`, `docs/research_protocol.md`, `docs/gpu_handoff.md`, `README.md` (setup).
