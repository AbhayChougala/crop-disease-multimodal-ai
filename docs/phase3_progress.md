# Phase 3 progress

Last updated: 2026-10-16

| Workstream | Status | Notes |
| --- | --- | --- |
| Phase 2 audit and Git safety | DONE | Git root verified; worktree was clean; no commit/push authorized. |
| Common lexical/semantic/hybrid retriever interface | DONE | TF-IDF retained; semantic FAISS and normalized hybrid retrieval implemented. |
| SentenceTransformers and FAISS runtime | REQUIRES EXTERNAL RESOURCE | Optional packages and explicit model download are not installed automatically. |
| Provider-neutral LLM orchestration and status | DONE | Mock/OpenAI/Ollama capabilities, health, structured output, and safe unavailable states. |
| Persistent diagnosis/general chat | DONE | SQLite sessions with bounded history, citations, deletion, and separate artifact archive. |
| Dedicated Ask CropAI interface | DONE | Two modes, citation drawer, language scaffold, retry/copy, and pipeline metadata. |
| HTML/PDF/JSON diagnosis reports | DONE | HTML/JSON always available; ReportLab PDF has explicit optional fallback. |
| RAG/grounding research evaluation | DONE | Retrieval metrics, human-review aggregator, sample labels, abstention/injection protocol. |
| Transport-level streaming | REQUIRES EXTERNAL RESOURCE | Provider interface fallback exists; progressive REST/WebSocket streaming was deprioritized. |
| Authoritative agricultural corpus | REQUIRES EXTERNAL RESOURCE | Repository manifest is empty; no sources or facts will be fabricated. |
| Real OpenAI/Ollama verification | REQUIRES EXTERNAL RESOURCE | Requires configured provider/model and, for OpenAI, an API key. |
| Real classifier and research results | REQUIRES EXTERNAL RESOURCE | Requires datasets, trained checkpoints, and reviewed experiments. |

No fabricated predictions, citations, agricultural claims, or metrics may be added during this phase.

Final local validation: Ruff passed; 32 Python tests passed; frontend typecheck/production build
passed; live TF-IDF + mock chat returned a manifest-backed citation; HTML/JSON/PDF reports generated;
the PDF was rendered and visually inspected. Interactive browser control was unavailable, so final
responsive browser QA remains a handoff item.
