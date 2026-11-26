#!/usr/bin/env bash
# Start the app.
#   ./run.sh              private: this machine / your Codespaces forwarded port
#   ./run.sh --expose     also create a temporary PUBLIC link (password protected); -expose works too
# PORT=8501 by default.
set -euo pipefail
cd "$(dirname "$0")"

usage() { sed -n '2,5p' run.sh | sed 's/^# \{0,1\}//'; }

expose=0
for arg in "$@"; do
  case "$arg" in
    -expose|--expose) expose=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$expose" -eq 1 ]; then
  exec uv run python scripts/expose.py --port "${PORT:-8501}"
fi

exec uv run streamlit run app/Home.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
