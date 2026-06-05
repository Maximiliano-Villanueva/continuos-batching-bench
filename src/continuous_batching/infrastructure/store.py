from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from continuous_batching.domain.models import ScenarioRun


class ResultStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS scenario_runs (
                run_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                concurrency_k INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                wall_time_s REAL,
                PRIMARY KEY (run_id, experiment_id, scenario_name, execution_mode, concurrency_k)
            );

            CREATE TABLE IF NOT EXISTS request_results (
                request_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                concurrency_k INTEGER NOT NULL,
                api_mode TEXT NOT NULL,
                prompt_class TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                position_in_wave INTEGER NOT NULL,
                success INTEGER NOT NULL,
                error TEXT,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                ttft_ms REAL,
                e2e_ms REAL NOT NULL,
                inter_token_ms_avg REAL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                sampled_at TEXT NOT NULL,
                memory_rss_mb REAL NOT NULL,
                memory_percent REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def save_run_metadata(self, run_id: str, config: dict[str, Any]) -> None:
        from continuous_batching.domain.models import utc_now

        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_metadata
            (run_id, config_json, created_at) VALUES (?, ?, ?)
            """,
            (run_id, json.dumps(config), utc_now().isoformat()),
        )
        self._conn.commit()

    def save_scenario_run(self, scenario: ScenarioRun) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO scenario_runs
            (run_id, experiment_id, scenario_name, execution_mode, concurrency_k,
             started_at, finished_at, wall_time_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario.run_id,
                scenario.experiment_id,
                scenario.scenario_name,
                scenario.execution_mode.value,
                scenario.concurrency_k,
                scenario.started_at.isoformat(),
                scenario.finished_at.isoformat() if scenario.finished_at else None,
                scenario.wall_time_s,
            ),
        )
        for result in scenario.request_results:
            row = result.to_row()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO request_results VALUES (
                    :request_id, :run_id, :experiment_id, :scenario_name,
                    :execution_mode, :concurrency_k, :api_mode, :prompt_class,
                    :prompt_id, :position_in_wave, :success, :error,
                    :input_tokens, :output_tokens, :ttft_ms, :e2e_ms,
                    :inter_token_ms_avg, :started_at, :finished_at
                )
                """,
                row,
            )
        for sample in scenario.system_samples:
            row = sample.to_row()
            self._conn.execute(
                """
                INSERT INTO system_samples
                (run_id, experiment_id, scenario_name, sampled_at, memory_rss_mb, memory_percent)
                VALUES (:run_id, :experiment_id, :scenario_name, :sampled_at,
                        :memory_rss_mb, :memory_percent)
                """,
                row,
            )
        self._conn.commit()

    def list_run_ids(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT run_id FROM run_metadata ORDER BY created_at DESC"
        )
        return [row["run_id"] for row in cur.fetchall()]

    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT config_json FROM run_metadata WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return json.loads(row["config_json"])

    def count_request_results(self, run_id: str | None = None) -> int:
        if run_id:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM request_results WHERE run_id = ?",
                (run_id,),
            )
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM request_results")
        return int(cur.fetchone()[0])

    def load_request_results(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            cur = self._conn.execute(
                "SELECT * FROM request_results WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            )
        else:
            cur = self._conn.execute("SELECT * FROM request_results ORDER BY started_at")
        return [dict(row) for row in cur.fetchall()]

    def load_scenario_runs(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            cur = self._conn.execute(
                "SELECT * FROM scenario_runs WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            )
        else:
            cur = self._conn.execute("SELECT * FROM scenario_runs ORDER BY started_at")
        return [dict(row) for row in cur.fetchall()]

    def load_system_samples(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            cur = self._conn.execute(
                "SELECT * FROM system_samples WHERE run_id = ? ORDER BY sampled_at",
                (run_id,),
            )
        else:
            cur = self._conn.execute("SELECT * FROM system_samples ORDER BY sampled_at")
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    @classmethod
    def from_results_dir(cls, results_dir: Path, run_id: str) -> ResultStore:
        return cls(results_dir / run_id / "benchmark.db")
