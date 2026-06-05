from __future__ import annotations

import os
from pathlib import Path


def find_repo_root() -> Path:
    """Locate project root whether running from source tree or an installed package."""
    env = os.environ.get("CONTINUOUS_BATCHING_ROOT")
    if env:
        root = Path(env).resolve()
        if _is_repo_root(root):
            return root

    cwd = Path.cwd().resolve()
    if _is_repo_root(cwd):
        return cwd

    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if _is_repo_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find repo root (expected configs/models.yaml). "
        "Run commands from the continuous-batching directory or set "
        "CONTINUOUS_BATCHING_ROOT=/path/to/continuous-batching"
    )


def _is_repo_root(path: Path) -> bool:
    return (path / "configs" / "models.yaml").is_file()
