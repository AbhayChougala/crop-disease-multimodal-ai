# Phase 3 continuation handoff

Phase 3 provides lexical/semantic/hybrid RAG abstractions, provider-neutral grounded generation,
durable two-mode chat, Ask CropAI, diagnosis exports, and evaluation tooling. Offline mock operation is
the tested default. No authoritative corpus, real classifier checkpoint, provider credential, or
publishable Phase 3 metric is present.

## Verified locally

- Mock diagnosis, durable diagnosis/general chat, history, deletion, and status APIs.
- Citation allow-listing, empty-retrieval behavior, and hostile retrieved-text prompt boundary.
- HTML/JSON report archive and optional ReportLab PDF path.
- Frontend TypeScript production build, including responsive Ask CropAI and report links.
- TF-IDF remains the baseline; semantic/FAISS and hybrid implementations load only explicit local
  indexes and validate model/dimension/corpus metadata.
- Benchmark outputs from the bundled fixture are forced to `SAMPLE_FIXTURE_ONLY`.

## External work required

1. Add reviewed, legally usable agricultural documents and complete manifest provenance.
2. Install semantic extras and explicitly download the selected embedding model; then build and test
   semantic and hybrid indexes.
3. Train/validate the vision checkpoints and segmentation model with real datasets.
4. Configure and test OpenAI or Ollama. Do not describe `CONFIGURED` as provider success.
5. Build a manually reviewed benchmark and complete blinded grounding/abstention evaluation.
6. Run responsive/accessibility browser QA on target devices.
7. Add transport-level token streaming only if time remains; REST is the Phase 3 reliable path.

## Start commands

```bash
cd "/Users/abhaychougala/Documents/Clg/4th year/7th sem/Project/crop_disease_multimodal_ai/backend"
source .venv/bin/activate
python -m uvicorn app.main:app --reload --reload-dir app --reload-dir src --port 8000
```

```bash
cd "/Users/abhaychougala/Documents/Clg/4th year/7th sem/Project/crop_disease_multimodal_ai/frontend"
npm install
npm run dev
```

If port 8000 is occupied, identify the exact process with `lsof -nP -iTCP:8000 -sTCP:LISTEN` and stop
that PID before restarting. Do not kill unrelated Python processes.

## Validation

```bash
source .venv/bin/activate
ruff check .
pytest
npm --prefix frontend run build
```

See `docs/rag_guide.md`, `docs/llm_setup.md`, `docs/report_generation.md`, and
`research/protocols/rag_evaluation.md` for exact subsystem commands.
