from pathlib import Path

from continuous_batching.infrastructure.store import ResultStore
from continuous_batching.reporting.export import export_run


def test_export_writes_html(tmp_path: Path):
    store = ResultStore(tmp_path / "bench.db")
    store.save_run_metadata("run1", {"model": "test"})
    store._conn.execute(
        """
        INSERT INTO request_results VALUES (
            'r1', 'run1', 'E4', 'short_prompt_k_sweep', 'concurrent', 1,
            'chat', 'short', 'short-1', 0, 1, NULL, 10, 20,
            5.0, 100.0, NULL,
            '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:10+00:00'
        )
        """
    )
    store._conn.execute(
        """
        INSERT INTO scenario_runs VALUES (
            'run1', 'E4', 'short_prompt_k_sweep', 'concurrent', 1,
            '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:10+00:00', 10.0
        )
        """
    )
    store._conn.commit()
    out = tmp_path / "out"
    export_run(store, out)
    store.close()
    assert (out / "summary.html").exists()
    assert (out / "summary.md").exists()
    assert "E4" in (out / "summary.html").read_text(encoding="utf-8")
