#!/usr/bin/env bash
# Start the app. Usage: ./run.sh   (PORT=8501 by default)
set -euo pipefail
cd "$(dirname "$0")"
exec uv run streamlit run app/Home.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
