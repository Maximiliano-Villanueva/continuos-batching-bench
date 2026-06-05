#!/usr/bin/env python3
"""Print shell exports for scripts/serve.sh from configs/models.yaml."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import yaml


def _split_serve_args(raw: str) -> tuple[list[str], str | None]:
    """Split CLI flags from an optional --speculative-config JSON blob."""
    text = raw.strip()
    if not text:
        return [], None

    spec_json: str | None = None
    match = re.search(r"--speculative-config\s+(\{.*\})\s*$", text)
    if match:
        spec_json = match.group(1)
        text = text[: match.start()].strip()

    return shlex.split(text), spec_json


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: resolve_serve_args.py <model_key> <on|off>")

    root = Path(__file__).resolve().parents[1]
    model_key, spec = sys.argv[1], sys.argv[2]
    data = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))
    entry = data["models"][model_key]
    hf_id = entry.get("mlx_hf_id") or entry["hf_id"]
    sp = entry.get("speculative") or {}
    profile = sp.get("when_enabled" if spec == "on" else "when_disabled") or {}
    base = str(entry.get("serve_base_args", "")).replace("\n", " ").strip()
    profile_args = str(profile.get("serve_args", "")).replace("\n", " ").strip()
    merged = " ".join(part for part in (base, profile_args) if part)
    reasoning = profile.get("reasoning_parser", "") if spec == "on" else ""

    argv, spec_json = _split_serve_args(merged)
    if spec_json:
        json.loads(spec_json)  # validate before passing to vllm

    print(f"export HF_ID={shlex.quote(hf_id)}")
    print(f"export REASONING_PARSER={shlex.quote(str(reasoning))}")
    if spec_json:
        print(f"export SPECULATIVE_CONFIG_JSON={shlex.quote(spec_json)}")
    else:
        print("export SPECULATIVE_CONFIG_JSON=")
    print("CMD_EXTRA=()")
    for arg in argv:
        print(f"CMD_EXTRA+=({shlex.quote(arg)})")


if __name__ == "__main__":
    main()
