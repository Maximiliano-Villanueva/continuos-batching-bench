#!/usr/bin/env bash
# Activate the project-local vllm-metal environment.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-vllm-metal"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing $VENV" >&2
  echo "Run from repo root: make install-vllm-metal" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
