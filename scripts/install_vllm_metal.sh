#!/usr/bin/env bash
# Install vllm-metal into .venv-vllm-metal/ inside this repo (not under $HOME).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-vllm-metal"
LIB_URL="https://raw.githubusercontent.com/vllm-project/vllm-metal/main/scripts/lib.sh"
INSTALL_URL="https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh"

echo "Installing vllm-metal into: $VENV"
echo "(Everything stays under $ROOT)"
echo ""

mkdir -p "$ROOT/scripts"
if [[ ! -f "$ROOT/scripts/lib.sh" ]]; then
  echo "Downloading vllm-metal install helpers..."
  curl -fsSL "$LIB_URL" -o "$ROOT/scripts/lib.sh"
fi

INSTALLER="$ROOT/.vllm-metal-install.sh"
curl -fsSL "$INSTALL_URL" -o "$INSTALLER"
chmod +x "$INSTALLER"

# Upstream install.sh uses PWD/.venv-vllm-metal when scripts/lib.sh exists next to it.
cd "$ROOT"
"$INSTALLER" "$@"
rm -f "$INSTALLER"

# Upstream install skips the vllm-metal wheel when scripts/lib.sh is local (it only
# runs `uv pip install .` from this repo). Install the Metal backend explicitly.
# shellcheck source=/dev/null
source "$VENV/bin/activate"
WHEEL_URL="$(
  python3 - <<'PY'
import json, urllib.request
data = json.load(urllib.request.urlopen(
    "https://api.github.com/repos/vllm-project/vllm-metal/releases/latest"
))
for asset in data.get("assets", []):
    name = asset.get("name", "")
    if name.endswith(".whl"):
        print(asset["browser_download_url"])
        break
PY
)"
if [[ -z "$WHEEL_URL" ]]; then
  echo "Could not resolve vllm-metal wheel URL" >&2
  exit 1
fi
echo ""
echo "Installing vllm-metal wheel: $WHEEL_URL"
uv pip install "$WHEEL_URL"

echo ""
echo "Done. Activate the server environment with:"
echo "  source $ROOT/.venv-vllm-metal/bin/activate"
echo "Or from the repo root:"
echo "  source scripts/activate-vllm.sh"
