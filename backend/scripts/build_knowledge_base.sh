#!/usr/bin/env bash
# Validate the knowledge manifest, chunk the documents, and build the RAG indexes.
#
# Usage (from anywhere):
#   backend/scripts/build_knowledge_base.sh            # TF-IDF + FAISS vector index
#   backend/scripts/build_knowledge_base.sh tfidf      # TF-IDF only (no embedding model)
#
# Inputs:  knowledge/source_manifest.csv, knowledge/source_docs/
# Outputs: data/vectorstore/{chunks.json,tfidf.joblib,semantic.faiss,semantic.metadata.json}
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-all}"
PYTHON=python
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
fi

rows=$(($(grep -cv '^[[:space:]]*$' knowledge/source_manifest.csv) - 1))
if [ "$rows" -lt 1 ]; then
  echo "knowledge/source_manifest.csv has no documents. Add files to knowledge/source_docs/" >&2
  echo "and one manifest row per file first (see docs/knowledge_base_guide.md)." >&2
  exit 1
fi

"$PYTHON" -m crop_ai.rag.validate \
  --manifest knowledge/source_manifest.csv --source-root knowledge/source_docs
"$PYTHON" -m crop_ai.rag.ingest
if [ "$MODE" = "tfidf" ]; then
  "$PYTHON" -m crop_ai.rag.build_index --mode tfidf
else
  # Downloads the embedding model (~90 MB) on the first run only; later runs use the local cache.
  "$PYTHON" -m crop_ai.rag.build_index --mode "$MODE" --allow-model-download
fi

echo
echo "Done. To use it, set in backend/.env:"
echo "  RAG_ENABLED=true"
if [ "$MODE" = "tfidf" ]; then
  echo "  RAG_RETRIEVER=tfidf"
else
  echo "  RAG_RETRIEVER=hybrid"
fi
echo "then restart the backend."
