from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SpeculativeProfile:
    method: str
    on_serve_args: str
    off_serve_args: str
    reasoning_parser: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ModelEntry:
    key: str
    display_name: str
    hf_id: str
    upstream_hf_id: str | None
    mlx_hf_id: str | None
    text_only: bool
    speculative: SpeculativeProfile

    def resolve_hf_id(self, use_mlx_weights: bool) -> str:
        if use_mlx_weights and self.mlx_hf_id:
            return self.mlx_hf_id
        return self.hf_id


@dataclass(frozen=True)
class ModelRegistry:
    models: dict[str, ModelEntry]
    default_model_key: str

    def get(self, key: str) -> ModelEntry:
        if key not in self.models:
            known = ", ".join(sorted(self.models))
            raise KeyError(f"Unknown model_key '{key}'. Known: {known}")
        return self.models[key]


def _parse_speculative(raw: dict[str, Any]) -> SpeculativeProfile:
    # YAML parses bare `on`/`off` as booleans — use when_enabled/when_disabled in config.
    enabled = raw.get("when_enabled") or raw.get("on") or {}
    disabled = raw.get("when_disabled") or raw.get("off") or {}
    if enabled is True:
        enabled = {}
    if disabled is False:
        disabled = {}
    return SpeculativeProfile(
        method=str(raw.get("method", "unknown")),
        on_serve_args=str(enabled.get("serve_args", "")).replace("\n", " ").strip(),
        off_serve_args=str(disabled.get("serve_args", "")).replace("\n", " ").strip(),
        reasoning_parser=enabled.get("reasoning_parser"),
        notes=str(raw.get("notes", "")).strip(),
    )


def load_model_registry(path: Path) -> ModelRegistry:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models: dict[str, ModelEntry] = {}
    for key, raw in (data.get("models") or {}).items():
        spec = _parse_speculative(raw.get("speculative") or {})
        models[key] = ModelEntry(
            key=key,
            display_name=str(raw.get("display_name", key)),
            hf_id=str(raw["hf_id"]),
            upstream_hf_id=raw.get("upstream_hf_id"),
            mlx_hf_id=raw.get("mlx_hf_id"),
            text_only=bool(raw.get("text_only", True)),
            speculative=spec,
        )

    if not models:
        raise ValueError(f"No models defined in {path}")

    return ModelRegistry(
        models=models,
        default_model_key=str(data.get("default_model_key", next(iter(models)))),
    )
