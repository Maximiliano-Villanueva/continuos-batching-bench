from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from continuous_batching.application.orchestrator import ExperimentOrchestrator
from continuous_batching.config_loader import load_run_config
from continuous_batching.domain.model_registry import load_model_registry
from continuous_batching.infrastructure.store import ResultStore
from continuous_batching.infrastructure.vllm_client import VllmOpenAIClient
from continuous_batching.paths import find_repo_root
from continuous_batching.reporting.export import export_run


def _parse_speculative(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized in ("on", "true", "1", "yes"):
        return True
    if normalized in ("off", "false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("speculative must be on or off")


async def _run_benchmark(args: argparse.Namespace) -> int:
    root = find_repo_root()
    model_path = Path(args.model_config) if args.model_config else root / "configs" / "model.yaml"
    exp_path = Path(args.config) if args.config else root / "configs" / "experiments.yaml"
    registry_path = root / "configs" / "models.yaml"
    prompts_dir = root / "scenarios" / "prompts"

    use_mlx: bool | None = None
    if getattr(args, "no_mlx", False):
        use_mlx = False
    elif getattr(args, "use_mlx", None):
        use_mlx = True

    config, _, registry = load_run_config(
        model_path,
        exp_path,
        registry_path,
        smoke=args.smoke,
        model_key_override=args.model,
        speculative_override=args.speculative,
        use_mlx_override=use_mlx,
    )
    entry = registry.get(config.model_key)

    print(f"Model key: {config.model_key} ({entry.display_name})")
    print(f"HF model id: {config.model}")
    print(f"Speculative tagging: {config.speculative_enabled} (must match server)")

    client = VllmOpenAIClient(
        base_url=config.base_url,
        model=config.model,
        timeout_seconds=config.timeout_seconds,
    )
    orchestrator = ExperimentOrchestrator(
        config=config,
        client=client,
        prompts_dir=prompts_dir,
        experiments_path=exp_path,
        resume_run_id=args.resume_run_id,
    )
    run_id = await orchestrator.run_all(
        only_experiment_ids=args.only_experiment or None,
        resume=bool(args.resume_run_id),
    )
    store = ResultStore.from_results_dir(Path(config.results_dir), run_id)
    export_run(store, Path(config.results_dir) / run_id)
    store.close()
    print(f"Benchmark complete. run_id={run_id}")
    print(f"Results: {config.results_dir}/{run_id}/")
    return 0


def _list_models(args: argparse.Namespace) -> int:
    root = find_repo_root()
    registry = load_model_registry(root / "configs" / "models.yaml")
    for key, entry in sorted(registry.models.items()):
        print(f"{key}: {entry.display_name}")
        print(f"  hf_id: {entry.hf_id}")
        if entry.mlx_hf_id:
            print(f"  mlx_hf_id: {entry.mlx_hf_id}")
        print(f"  speculative: {entry.speculative.method}")
        if entry.speculative.notes:
            print(f"  note: {entry.speculative.notes}")
    return 0


def _export(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    if args.run_id:
        run_dirs = [results_dir / args.run_id]
    else:
        candidates = sorted(
            [p for p in results_dir.iterdir() if p.is_dir() and (p / "benchmark.db").is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        run_dirs = []
        for run_dir in candidates:
            store = ResultStore(run_dir / "benchmark.db")
            if store.count_request_results():
                run_dirs.append(run_dir)
            store.close()
    if not run_dirs:
        print("No results found.", file=sys.stderr)
        return 1
    for run_dir in run_dirs[:1]:
        db = run_dir / "benchmark.db"
        store = ResultStore(db)
        if not store.count_request_results():
            print(f"No request data in {run_dir}", file=sys.stderr)
            store.close()
            return 1
        export_run(store, run_dir)
        store.close()
        print(f"Exported {run_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous batching benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run benchmark suite")
    run_p.add_argument("--config", default=None, help="experiments.yaml path")
    run_p.add_argument("--model-config", default=None, help="model.yaml path")
    run_p.add_argument(
        "--model",
        default=None,
        choices=["gemma-4-e4b", "qwen3.5-4b"],
        help="model key from configs/models.yaml",
    )
    run_p.add_argument(
        "--speculative",
        type=_parse_speculative,
        default=None,
        help="tag run: on/off (must match how vllm serve was started)",
    )
    run_p.add_argument(
        "--use-mlx",
        action="store_true",
        default=None,
        help="use mlx_hf_id when available",
    )
    run_p.add_argument(
        "--no-mlx",
        action="store_true",
        help="use upstream hf_id instead of MLX quant",
    )
    run_p.add_argument(
        "--smoke",
        action="store_true",
        help="run smoke profile (minimal requests, connectivity check only)",
    )
    run_p.add_argument(
        "--resume-run-id",
        default=None,
        help="append to an existing results/<run_id>/benchmark.db (skips finished scenarios)",
    )
    run_p.add_argument(
        "--only-experiment",
        action="append",
        default=[],
        metavar="EXPERIMENT_ID",
        help="run only these experiments (e.g. E7); may be repeated",
    )
    run_p.set_defaults(func=lambda a: asyncio.run(_run_benchmark(a)))

    sub.add_parser("models", help="List registered models").set_defaults(func=_list_models)

    exp_p = sub.add_parser("export", help="Export CSV/MD for a run")
    exp_p.add_argument("--results-dir", default="results")
    exp_p.add_argument("--run-id", default=None)
    exp_p.set_defaults(func=_export)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
