# Implementation status

Last updated: 2026-10-15

## Implemented

- Project rules, safe ignore patterns, and environment template
- FastAPI application and typed response schemas
- Health and system-capabilities endpoints
- Upload validation, preprocessing preview, and mock diagnosis
- Deterministic mock classifier with explicit `is_mock=true`
- Modular OpenCV preprocessing and heuristic leaf mask
- Independent experimental lesion color mask and affected-area calculation
- Data-URL artifacts and measured stage timings
- Mock advisory and chat responses that avoid unsupported agricultural claims
- Feedback persistence in ignored runtime storage
- Realtime WebSocket inference interface
- React upload, results, research, about, and live-camera pages
- Unit/integration test suite

## Not implemented yet

- Real checkpoint loading or validated classifier output
- Training/evaluation pipeline and dataset manifests
- Grad-CAM generation
- Agricultural document ingestion, embeddings, or retrieval
- OpenAI/local LLM providers
- Diagnosis session persistence for context-aware chat
- Validated experimental results or metrics

## Scientific status

The current system is a software prototype in demo mode. Mock predictions must not be copied into research results. Severity categories are operational software categories, not agricultural standards.

## Validation on 2026-10-15

- `ruff check .`: passed
- `pytest`: passed (11 tests)
- `npm run build`: passed
- Live health and capabilities requests: passed
- Live multipart diagnosis request: passed in mock mode
- Live realtime WebSocket request: passed in mock mode
- Pixel-level browser QA: not completed because the in-app preview connection failed during setup; the frontend was verified by TypeScript and production build checks instead
