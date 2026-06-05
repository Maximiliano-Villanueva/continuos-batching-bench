#!/usr/bin/env python3
"""Print expected request counts for a benchmark profile (no server required)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _scenario_count(exp: dict, root: dict, profile: dict) -> list[tuple[str, str, int, int]]:
    """Return (exp_id, scenario_name, wave_size, scenarios) helper rows."""
    reps = int(profile.get("repetitions", root.get("repetitions", 15)))
    rows: list[tuple[str, str, int, int]] = []

    if exp["id"] == "E1":
        for wave in exp.get("waves", []):
            if wave.get("wave_size"):
                w = int(wave["wave_size"])
            elif wave["name"] == "long_short_mix":
                w = 5 * int(wave.get("wave_cycles", profile.get("e1_mix_cycles", 5)))
            else:
                w = len(wave.get("prompt_classes", [1]))
            for _ in exp.get("modes", []):
                rows.append((exp["id"], wave["name"], w, reps))
    elif exp["id"] == "E2":
        w = 4 * int(profile.get("e2_wave_cycles", 5))
        for order in exp.get("orders", []):
            rows.append((exp["id"], order["name"], w, reps))
    elif exp["id"] == "E3":
        w = int(profile.get("e3_wave_size", 8))
        for _ in exp.get("modes", []):
            for pc in exp.get("prompt_classes", []):
                rows.append((exp["id"], str(pc), w, reps))
    elif exp["id"] == "E4":
        w = int(profile.get("e4_wave_size", 48))
        for k in profile.get("concurrency_sweep", root.get("concurrency_sweep", [])):
            rows.append((exp["id"], f"k{k}", w, reps))
    elif exp["id"] == "E5":
        w = 1 + int(profile.get("e5_short_mix_count", 12))
        for _ in exp.get("modes", []):
            rows.append((exp["id"], "short_long_output", w, reps))
    elif exp["id"] == "E6":
        w = int(profile.get("e6_wave_size", 32))
        for api in exp.get("api_modes", []):
            rows.append((exp["id"], f"api_{api}", w, reps))
    elif exp["id"] == "E7":
        w = int(profile.get("e7_wave_size", 24))
        k_vals = profile.get("e7_k_values", root.get("e7_k_values", [1, 4, 8]))
        for mode in exp.get("modes", []):
            for k in k_vals:
                if mode == "sequential" and int(k) != 1:
                    continue
                rows.append((exp["id"], f"{mode}_k{k}", w, reps))
    return rows


def main() -> None:
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "rigorous"
    data = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    profile = (data.get("profiles") or {}).get(profile_name, data)
    warmup = int(profile.get("warmup_requests", data.get("warmup_requests", 0)))

    total_measured = 0
    scenarios = 0
    print(f"Profile: {profile_name}\n")
    print(f"{'Experiment':<6} {'Scenario':<28} {'Wave':>6} {'Reps':>6} {'Measured':>10}")
    print("-" * 62)

    for exp in data.get("experiments", []):
        if not exp.get("enabled", True):
            continue
        for exp_id, name, wave, reps in _scenario_count(exp, data, profile):
            measured = wave * reps
            total_measured += measured
            scenarios += 1
            print(f"{exp_id:<6} {name:<28} {wave:>6} {reps:>6} {measured:>10}")

    warmups = scenarios * warmup
    print("-" * 62)
    print(f"Scenarios: {scenarios}")
    print(f"Measured requests (stored): {total_measured:,}")
    print(f"Warmup requests (not stored): {warmups:,}")
    print(f"Total HTTP calls: {total_measured + warmups:,}")
    print()
    print("Tip: run `python scripts/build_prompts.py` before rigorous runs for ~8k-token long prompts.")


if __name__ == "__main__":
    main()
