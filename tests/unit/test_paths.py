from continuous_batching.paths import find_repo_root


def test_find_repo_root(repo_root):
    assert find_repo_root() == repo_root or (repo_root / "configs" / "models.yaml").is_file()
