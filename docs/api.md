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
| POST | `/chat` | Diagnosis-aware mock or grounded configured provider |
| GET | `/chat/{session_id}/history` | Bounded in-memory session history |
| POST | `/feedback` | Implemented |
| GET | `/research/summary` | Implemented; never invents metrics |
| WS | `/realtime/ws` | Throttled client-driven demo or real inference |

Upload endpoints accept JPEG, PNG, or WebP multipart files up to `MAX_UPLOAD_MB`. Interactive OpenAPI documentation is served at `/docs`.
Real inference returns HTTP 503 when its configured checkpoint cannot load; it never changes to demo
mode implicitly. Knowledge/provider failures return explicit unavailable states without exposing secrets
or stack traces.
