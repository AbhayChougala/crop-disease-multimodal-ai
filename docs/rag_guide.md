# RAG guide

Run the commands below from `backend/` with its virtual environment active.

CropAI has three interchangeable local retrievers: TF-IDF (`tfidf`), SentenceTransformers + FAISS
(`semantic`), and their normalized weighted merger (`hybrid`). TF-IDF remains the lexical research
baseline. Semantic dependencies and model downloads are optional and never happen at API startup.

## Seed corpus

`scripts/fetch_knowledge_sources.py` downloads ~208 extension publications: UC IPM Pest Management
Guidelines (California), TNAU Agritech Portal (India; disease pages are discovered from each crop
index), NC State, Ohio State, UF/IFAS, WSU, Delaware, Maine and Oklahoma State. It covers every
PlantVillage crop plus chilli, brinjal, bhendi, onion, cabbage, banana and mango. Page boilerplate is
stripped, and `knowledge/source_manifest.csv` is written with dates and SHA-256 hashes. Use
`--list` to preview and `--refresh` to re-download. Some sites (UMN, UMass, Clemson, Missouri, WVU,
PlantwisePlus) block automated downloads and are intentionally not fetched.

## One-command build

```bash
python -m pip install -e '.[semantic-rag]'   # or '.[rag]' for TF-IDF only
scripts/build_knowledge_base.sh             # validate -> ingest -> TF-IDF + FAISS
scripts/build_knowledge_base.sh tfidf       # TF-IDF only
```

Then set `RAG_ENABLED=true` and `RAG_RETRIEVER=hybrid` (or `tfidf`) in `.env` and restart the API.
The steps it runs are listed below.

## Approved sources and ingestion

Only files explicitly declared in `knowledge/source_manifest.csv` can be ingested. Supported formats
are PDF, TXT, Markdown, and HTML. The manifest preserves document ID, title, organization, author,
publication date, source URL, crop, disease, source type, usage/license note, local path, and optional
SHA-256. Source files and generated indexes are Git-ignored.

```bash
source .venv/bin/activate
python -m crop_ai.rag.validate \
  --manifest knowledge/source_manifest.csv --source-root knowledge/source_docs
python -m crop_ai.rag.ingest
python -m crop_ai.rag.build_index --mode tfidf
python -m crop_ai.rag.retriever --query "tomato early blight uncertainty"
```

The committed manifest is intentionally empty. Add only reviewed, legally usable documents; never
substitute the test fixture for an authoritative corpus.

## Semantic and hybrid indexes

Install optional dependencies, review the configured model, and explicitly permit its first download:

```bash
python -m pip install -e '.[semantic-rag]'
python -m crop_ai.rag.build_index --mode semantic --allow-model-download
python -m crop_ai.rag.build_index --mode all --allow-model-download
```

Subsequent API loads use local files only. FAISS metadata records model ID, vector dimensions, corpus
hash, manifest hash, creation time, and document/chunk counts. Loading rejects a model, dimension, or
corpus mismatch. Chunk metadata remains in JSON beside the binary FAISS index.

```dotenv
RAG_ENABLED=true
RAG_RETRIEVER=hybrid
RAG_TOP_K=5
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HYBRID_LEXICAL_WEIGHT=0.5
HYBRID_SEMANTIC_WEIGHT=0.5
```

Crop and predicted disease are query hints by default. Set `RAG_STRICT_DIAGNOSIS_FILTERS=true` only
for an experiment requiring hard filters. Every citation is derived from indexed chunk metadata.

## Evaluation

```bash
python -m crop_ai.rag.evaluate \
  --retriever tfidf \
  --cases research/protocols/sample_rag_benchmark.csv \
  --output research/results/sample_tfidf_rag.json \
  --table-output research/tables/sample_tfidf_rag.csv
```

The sample benchmark is synthetic and output is labelled `SAMPLE_FIXTURE_ONLY`. Use a manually
reviewed benchmark for research claims. See `research/protocols/rag_evaluation.md`.
