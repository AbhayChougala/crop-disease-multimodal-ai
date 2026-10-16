#!/usr/bin/env bash
set -euo pipefail
# Run from backend/ so `app` imports and relative artifact paths resolve.
cd "$(dirname "$0")/.."
PYTHON=python
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
fi
# Reload only on code changes; data, indexes and runtime files under backend/ must not restart it.
exec "$PYTHON" -m uvicorn app.main:app --reload --reload-dir app --reload-dir src \
  --port "${API_PORT:-8000}"
