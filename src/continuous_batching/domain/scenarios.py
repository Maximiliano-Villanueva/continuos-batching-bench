from __future__ import annotations

import json
from pathlib import Path

from continuous_batching.domain.models import PromptClass, PromptSpec


def load_prompts(prompts_dir: Path) -> dict[str, PromptSpec]:
    prompts: dict[str, PromptSpec] = {}
    for path in sorted(prompts_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                spec = PromptSpec(
                    id=row["id"],
                    prompt_class=PromptClass(row["prompt_class"]),
                    text=row["text"],
                    max_tokens=row.get("max_tokens"),
                    metadata=row.get("metadata", {}),
                )
                prompts[spec.id] = spec
    if not prompts:
        raise FileNotFoundError(f"No prompts found in {prompts_dir}")
    return prompts


def pick_prompts(
    prompts: dict[str, PromptSpec],
    classes: list[str | PromptClass],
) -> list[PromptSpec]:
    by_class: dict[PromptClass, list[PromptSpec]] = {}
    for spec in prompts.values():
        by_class.setdefault(spec.prompt_class, []).append(spec)

    result: list[PromptSpec] = []
    for i, cls in enumerate(classes):
        pc = PromptClass(cls) if isinstance(cls, str) else cls
        pool = by_class.get(pc)
        if not pool:
            raise ValueError(f"No prompts for class {pc}")
        result.append(pool[i % len(pool)])
    return result


def _clone_prompt(spec: PromptSpec, suffix: str) -> PromptSpec:
    return PromptSpec(
        id=f"{spec.id}-{suffix}",
        prompt_class=spec.prompt_class,
        text=spec.text,
        max_tokens=spec.max_tokens,
        metadata=dict(spec.metadata),
    )


def build_wave(
    prompts: dict[str, PromptSpec],
    classes: list[str | PromptClass],
    *,
    cycles: int = 1,
    target_size: int | None = None,
) -> list[PromptSpec]:
    """Expand a prompt pattern to a larger wave for statistically meaningful runs."""
    base = pick_prompts(prompts, classes)
    if cycles < 1:
        cycles = 1

    desired = target_size if target_size is not None else len(base) * cycles
    if desired < 1:
        desired = len(base)

    wave: list[PromptSpec] = []
    cycle_idx = 0
    while len(wave) < desired:
        for pos, spec in enumerate(base):
            wave.append(_clone_prompt(spec, f"c{cycle_idx}p{pos}"))
            if len(wave) >= desired:
                break
        cycle_idx += 1
    return wave
