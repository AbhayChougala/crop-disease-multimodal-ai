#!/usr/bin/env bash
set -euo pipefail
uvicorn backend.app.main:app --reload --port "${API_PORT:-8000}"
