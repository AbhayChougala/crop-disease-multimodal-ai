# API

Base path: `/api/v1`

| Method | Path | Status |
| --- | --- | --- |
| GET | `/health` | Implemented |
| GET | `/system/capabilities` | Implemented |
| GET | `/models` | Supported architectures and loaded-checkpoint metadata |
| POST | `/preprocess/preview` | Implemented |
| POST | `/diagnose` | Demo or configured real checkpoint; no silent fallback |
| GET | `/knowledge/search` | Local source-preserving retrieval when enabled |
| GET | `/knowledge/sources` | Indexed source metadata |
| GET | `/rag/status` | Retriever and index status |
| POST | `/rag/rebuild` | Fixed-path local rebuild; development only |
| GET | `/llm/status` | Provider capabilities and availability |
| POST | `/chat/sessions` | Create persistent general/diagnosis session |
| GET/DELETE | `/chat/sessions/{session_id}` | Load or clear a session |
| GET/POST | `/chat/sessions/{session_id}/messages` | Bounded durable history / ask question |
| POST | `/chat/general` | Create general session and answer |
| POST | `/chat/diagnosis/{diagnosis_id}` | Diagnosis-context answer |
| POST | `/chat` | Backward-compatible diagnosis/general session message |
| GET | `/reports/diagnosis/{id}.html` | Portable diagnosis report |
| GET | `/reports/diagnosis/{id}.pdf` | Optional ReportLab report |
| GET | `/reports/diagnosis/{id}.json` | Typed evidence export |
| POST | `/feedback` | Implemented |
| GET | `/research/summary` | Implemented; never invents metrics |
| WS | `/realtime/ws` | Throttled client-driven demo or real inference |

Upload endpoints accept JPEG, PNG, or WebP multipart files up to `MAX_UPLOAD_MB`. Interactive OpenAPI documentation is served at `/docs`.
Real inference returns HTTP 503 when its configured checkpoint cannot load; it never changes to demo
mode implicitly. Knowledge/provider failures return explicit unavailable states without exposing secrets
or stack traces.
