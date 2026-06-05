#!/usr/bin/env bash
# Optional GPU power sampling on macOS (requires sudo).
# Usage: sudo ./scripts/sample_powermetrics.sh 60 > gpu_power.log
set -euo pipefail
DURATION="${1:-30}"
powermetrics --samplers gpu_power -i 1000 -n "$DURATION"
