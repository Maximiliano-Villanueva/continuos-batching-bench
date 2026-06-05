#!/usr/bin/env bash
# Start vllm-metal for a registered model with speculative decoding on or off.
#
# Usage:
#   ./scripts/serve.sh gemma-4-e4b off
#   ./scripts/serve.sh gemma-4-e4b on
#
# Uses .venv-vllm-metal/ in this repo (run: make install-vllm-metal first).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-vllm-metal"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "vllm-metal is not installed in $VENV" >&2
  echo "Run: make install-vllm-metal" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

# vLLM can exhaust the default fd limit under concurrent load.
ulimit -n 65536 2>/dev/null || true

MODEL_KEY="${1:-gemma-4-e4b}"
SPEC="${2:-off}"

eval "$(python3 "$ROOT/scripts/resolve_serve_args.py" "$MODEL_KEY" "$SPEC")"

echo "Serving: $HF_ID (model_key=$MODEL_KEY, speculative=$SPEC)"
echo "Venv: $VENV"
CMD=(vllm serve "$HF_ID" --host 127.0.0.1 --port 8000)
if [[ -n "${REASONING_PARSER:-}" ]]; then
  CMD+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ ${#CMD_EXTRA[@]} -gt 0 ]]; then
  CMD+=("${CMD_EXTRA[@]}")
fi
if [[ -n "${SPECULATIVE_CONFIG_JSON:-}" ]]; then
  CMD+=(--speculative-config "$SPECULATIVE_CONFIG_JSON")
fi
exec "${CMD[@]}"
