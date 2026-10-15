# Implementation status

Last updated: 2026-10-15

## Implemented

- Three torchvision classifier factories, training/evaluation CLIs, safe checkpoints, top-k,
  calibration metadata, abstention, and one-time runtime loading
- Manifest preparation/validation, group-aware splitting, fingerprints, duplicate/leakage reports
- OpenCV stages, class-specific Grad-CAM, U-Net training/inference, mask metrics, and explicit
  experimental/validated severity states
- Typed multimodal evidence with model, image, preprocessing, uncertainty, and provenance fields
- Local source ingestion, chunking, TF-IDF persistence/search, citations, and retrieval evaluation
- Mock, OpenAI Responses API, and Ollama LLM providers with grounded citation validation
- Diagnosis sessions, context-aware chat, knowledge search, WebSocket inference, and capability APIs
- React diagnosis artifacts/chat, live camera, and API-backed research protocol/results page
- Experiment metadata and genuine-result aggregation that leaves missing work pending
- Unit, integration, ML smoke, lint, and frontend production-build validation

## External-input or follow-up work

- No agricultural dataset, trained weights, validated lesion masks, authoritative knowledge corpus,
  provider credentials, or genuine research metrics are included.
- Sentence-transformer/FAISS semantic retrieval, real-provider streaming, durable session storage, and
  direct orchestration/plotting of every controlled experiment remain follow-up work.

## Scientific status

The current system is a software prototype in demo mode. Mock predictions must not be copied into research results. Severity categories are operational software categories, not agricultural standards.

## Validation on 2026-10-15

- `ruff check .`: passed
- `pytest -q`: passed (24 passed, 3 skipped because optional torch/scikit-learn are absent from the app venv)
- `PYTHONPATH=.:src python3 scripts/ml_smoke_test.py`: passed with global PyTorch 2.6 CPU
- `PYTHONPATH=.:src python3 scripts/training_smoke_test.py`: passed one CPU epoch; synthetic metrics were discarded
- `ruff check .`: passed
- `npm --prefix frontend run build`: passed
- The integration suite covers health, capabilities, multipart diagnosis, context chat, and WebSocket
  inference in explicit demo mode.
