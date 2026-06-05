from pathlib import Path

import pytest

from continuous_batching.config_loader import load_run_config
from continuous_batching.domain.model_registry import load_model_registry


def test_registry_models(repo_root: Path):
    reg = load_model_registry(repo_root / "configs" / "models.yaml")
    assert "gemma-4-e4b" in reg.models
    assert "qwen3.5-4b" in reg.models
    gemma = reg.get("gemma-4-e4b")
    assert "gemma-4" in gemma.hf_id
    qwen = reg.get("qwen3.5-4b")
    assert qwen.resolve_hf_id(True) == qwen.mlx_hf_id
    assert qwen.resolve_hf_id(False) == qwen.hf_id


def test_load_run_config_resolves_qwen_mlx(repo_root: Path):
    config, _, registry = load_run_config(
        repo_root / "configs" / "model.yaml",
        repo_root / "configs" / "experiments.yaml",
        repo_root / "configs" / "models.yaml",
        model_key_override="qwen3.5-4b",
        use_mlx_override=True,
    )
    entry = registry.get("qwen3.5-4b")
    assert config.model == entry.mlx_hf_id
    assert config.model_key == "qwen3.5-4b"


def test_speculative_profile(repo_root: Path):
    qwen = load_model_registry(repo_root / "configs" / "models.yaml").get("qwen3.5-4b")
    assert qwen.speculative.method == "mtp"
    assert "qwen3_next_mtp" in qwen.speculative.on_serve_args
    gemma = load_model_registry(repo_root / "configs" / "models.yaml").get("gemma-4-e4b")
    assert gemma.speculative.method == "draft_model"


def test_unknown_model_raises(repo_root: Path):
    reg = load_model_registry(repo_root / "configs" / "models.yaml")
    with pytest.raises(KeyError):
        reg.get("does-not-exist")
