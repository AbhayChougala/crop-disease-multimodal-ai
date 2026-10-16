# Chatbot architecture

Ask CropAI has two modes sharing one orchestration and citation policy.

```text
general question ─────────────┐
                             ├─> retriever ─> approved chunks ─> provider ─> validated citations
diagnosis evidence + question ┘
```

Diagnosis mode keeps typed visual evidence outside message history. General mode explicitly records
that no image evidence exists. Recent history is capped by `CHAT_MAX_HISTORY_MESSAGES` and loaded
from SQLite. The database stores text, citation references, timings, provider/model metadata, and
session timestamps—never images, API keys, or environment values. Full diagnosis artifacts are kept
in the ignored runtime diagnosis archive for report generation.

Retrieved text is enclosed as untrusted data. System policy prohibits following document
instructions, exposing secrets/prompts, inventing visual observations, citations, pesticides, doses,
or schedules. Real providers are not called when retrieval is empty. Returned citation IDs are
intersected with retrieved IDs before reaching the UI.

REST endpoints cover session create/load/delete, messages, diagnosis/general convenience calls,
provider status, and retrieval status. Provider interfaces expose a stream method, but Phase 3 uses
reliable REST responses; progressive network streaming is intentionally deferred.

The frontend shows observable stages and metadata only. It does not expose hidden reasoning.
