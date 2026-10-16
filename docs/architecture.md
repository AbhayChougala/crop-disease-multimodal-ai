# Architecture

The system separates HTTP/UI concerns from reusable research code.

```text
React upload or camera
        |
        v
FastAPI routes -> diagnosis/realtime services
        |
        +-> OpenCV decode and configurable preprocessing
        +-> leaf mask -> independent lesion mask -> explicit severity status
        +-> one-time model manager (explicit demo or safe state-dict checkpoint)
        +-> class-specific Grad-CAM and typed provenance-rich evidence
        +-> lexical / semantic / hybrid retrieval -> mock/OpenAI/Ollama provider
        +-> SQLite chat metadata + separate diagnosis artifact archive
        +-> HTML / optional PDF / JSON report renderers
        |
        v
Typed response with artifacts, sources, timings, warnings, and is_mock
```

The dedicated Ask CropAI page supports general and diagnosis-aware modes. Retrieved chunks are
untrusted data; provider output can only cite IDs present in the retrieval result. The application
abstains before calling a real provider when no source is available.

`backend/src/crop_ai` contains framework-independent CV, model, evidence, retrieval, and generation code. `backend/app` owns transport, validation, configuration, and orchestration. `frontend` consumes the API contract and never receives credentials.

Classification training/evaluation, U-Net segmentation, RAG indexing, and experiment aggregation run as
offline CLIs. Generated datasets, checkpoints, vector indexes, and results stay outside Git.

Grad-CAM explains classifier influence. It remains separate from the experimental lesion mask and is
never used as severity ground truth. Real model mode fails closed; it does not silently produce demo
predictions when a checkpoint is missing or invalid.

## Repository layout

```text
backend/    Python project root: pyproject.toml, .env, .venv
  app/        FastAPI transport, config, services (import name `app`)
  src/crop_ai/  CV, models, RAG, LLM, reports (research library)
  training/ scripts/ tests/ configs/
  data/ artifacts/ knowledge/ research/   (generated content is git-ignored)
frontend/   React + Vite app (npm commands run here)
docs/       documentation
```

Every Python command in these docs (pip, pytest, ruff, `python -m crop_ai...`, uvicorn) runs from
`backend/`; relative paths such as `configs/...`, `data/...` and `artifacts/...` are relative to it.
