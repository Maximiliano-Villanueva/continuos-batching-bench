from pathlib import Path

import pytest

from continuous_batching.domain.models import PromptClass
from continuous_batching.domain.scenarios import load_prompts, pick_prompts


def test_load_prompts(repo_root: Path):
    prompts = load_prompts(repo_root / "scenarios" / "prompts")
    assert "short-1" in prompts
    assert prompts["short-1"].prompt_class == PromptClass.SHORT


def test_pick_prompts(repo_root: Path):
    prompts = load_prompts(repo_root / "scenarios" / "prompts")
    picked = pick_prompts(prompts, ["short", "long", "short"])
    assert picked[0].prompt_class == PromptClass.SHORT
    assert picked[1].prompt_class == PromptClass.LONG


def test_pick_missing_class_raises(repo_root: Path):
    prompts = load_prompts(repo_root / "scenarios" / "prompts")
    with pytest.raises(ValueError):
        pick_prompts(prompts, ["nonexistent"])
