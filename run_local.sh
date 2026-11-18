#!/usr/bin/env bash
# One-command start (macOS / Linux): sets up Python, checks Ollama, pulls the model that fits
# this machine, opens the local sales agent. Needs internet ONCE (installs + model download).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || { echo "→ creating .venv"; $PY -m venv .venv; }
. .venv/bin/activate
if ! python -c "import sellbird, psutil" 2>/dev/null; then
  echo "→ installing Sellbird (first run only)"
  pip install -q -e . -r requirements-local.txt
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "✗ Ollama is not installed. Get it from https://ollama.com (free), then run this again."; exit 1
fi
if ! curl -s -m 3 http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "→ starting ollama serve in the background"; (ollama serve >/dev/null 2>&1 &); sleep 3
fi
MODEL=${SELLBIRD_LOCAL_MODEL:-$(python -c "from sellbird.local_sales.hardware import recommend_model as r; print(r()['recommended_model'])")}
echo "→ model for this machine: $MODEL"
python scripts/verify_local.py embeddings >/dev/null 2>&1 || {
  echo "→ caching the embedding model once (needs internet)"
  python scripts/verify_local.py embeddings --download >/dev/null 2>&1 || echo "  (could not cache it: scoring will use the labelled TF-IDF fallback)"; }
ollama list | awk 'NR>1{print $1}' | grep -qx "$MODEL" || ollama pull "$MODEL"
export SELLBIRD_LOCAL_MODEL="$MODEL"
echo "→ open http://127.0.0.1:8765  (Ctrl+C to stop)"
exec python -m sellbird.local_sales serve
