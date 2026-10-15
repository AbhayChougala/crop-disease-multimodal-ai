# RAG and LLM guide

Only approved local documents listed in `knowledge/source_manifest.csv` are ingested. Record title,
organization, source URL, date, crop/disease filters, usage/license notes, local path, and an optional
SHA-256 version hash. Do not scrape or commit copyrighted documents.

```bash
python -m crop_ai.rag.validate \
  --manifest knowledge/source_manifest.csv --source-root knowledge/source_docs
python -m crop_ai.rag.ingest
python -m crop_ai.rag.retriever --build
python -m crop_ai.rag.retriever --query "tomato early blight uncertainty"
```

The default index is local TF-IDF and requires no paid API or model download. It preserves document,
chunk, organization, crop, disease, and PDF page metadata. Treat `.joblib` indexes as trusted local
generated files; never load an index from an untrusted source. Sentence-transformer/FAISS retrieval is
a documented future extension, not silently downloaded today.

Enable runtime retrieval only after building an index:

```dotenv
RAG_ENABLED=true
RAG_INDEX_PATH=data/vectorstore/tfidf.joblib
LLM_PROVIDER=openai
LLM_MODEL=<configured-model>
LLM_API_KEY=<secret>
```

For Ollama, use `LLM_PROVIDER=ollama`, `LLM_BASE_URL=http://localhost:11434`, and a local model name.
The OpenAI implementation uses the Responses API structured-output parser. Both real providers refuse
to generate grounded advice when retrieval is empty, validate returned citation IDs, treat retrieved
text as untrusted, and never infer visual symptoms absent from structured evidence.

Evaluate retrieval with a committed JSON query set containing `query` and `expected_document_id`:

```bash
python -m crop_ai.rag.evaluate --index data/vectorstore/tfidf.joblib \
  --cases research/protocols/retrieval_cases.json \
  --output research/results/raw/llm_grounding/retrieval.json
```
