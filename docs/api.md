# API

Base path: `/api/v1`

| Method | Path | Status |
| --- | --- | --- |
| GET | `/health` | Implemented |
| GET | `/system/capabilities` | Implemented |
| POST | `/preprocess/preview` | Implemented |
| POST | `/diagnose` | Implemented in mock model mode |
| POST | `/chat` | Mock interface implemented |
| POST | `/feedback` | Implemented |
| GET | `/research/summary` | Implemented; never invents metrics |
| WS | `/realtime/ws` | Implemented in mock model mode |

Upload endpoints accept JPEG, PNG, or WebP multipart files up to `MAX_UPLOAD_MB`. Interactive OpenAPI documentation is served at `/docs`.
