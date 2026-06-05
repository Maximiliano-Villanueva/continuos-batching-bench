#!/usr/bin/env python3
"""Verify vLLM OpenAI-compatible server is reachable."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "configs" / "model.yaml"
    with model_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg["base_url"].rstrip("/")
    if base.endswith("/v1"):
        url = base[: -len("/v1")] + "/v1/models"
    else:
        url = base + "/models"

    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("id") for m in data.get("data", [])]
        print(f"OK: server at {url}")
        print(f"Models: {models}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot reach vLLM at {url}: {exc}", file=sys.stderr)
        print(
            "Start the server in another terminal: ./scripts/serve.sh <model> off",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
