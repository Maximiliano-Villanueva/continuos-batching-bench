#!/usr/bin/env python3
"""Generate expanded prompt fixtures (optional; keeps repo small by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _long_text(target_chars: int = 8000) -> str:
    base = (
        "The following is filler context for a long-prompt benchmark. "
        "Continuous batching schedules prefills and decodes dynamically. "
    )
    parts = []
    while sum(len(p) for p in parts) < target_chars:
        parts.append(base)
    return "".join(parts)[:target_chars]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("scenarios/prompts/generated.jsonl"))
    args = parser.parse_args()

    rows = [
        {
            "id": "long-generated",
            "prompt_class": "long",
            "text": "Analyze and summarize:\n\n" + _long_text(8000),
        }
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
