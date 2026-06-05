from continuous_batching.domain.scenarios import build_wave, load_prompts


def test_build_wave_target_size(repo_root):
    prompts = load_prompts(repo_root / "scenarios" / "prompts")
    wave = build_wave(prompts, ["short"], target_size=10)
    assert len(wave) == 10
    assert len({p.id for p in wave}) == 10


def test_build_wave_cycles(repo_root):
    prompts = load_prompts(repo_root / "scenarios" / "prompts")
    wave = build_wave(prompts, ["long", "short"], cycles=3)
    assert len(wave) == 6
