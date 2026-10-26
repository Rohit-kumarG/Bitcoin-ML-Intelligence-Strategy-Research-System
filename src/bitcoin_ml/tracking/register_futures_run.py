"""Register current Futures research artifacts and metrics in a local SQLite database."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("artifacts/registry/bitcoin_ml_experiments.sqlite3")
INPUT_FILES = [
    Path("data/raw/btcusdt_usdm_1h.csv"),
    Path("data/raw/btcusdt_usdm_funding.csv"),
    Path("data/processed/btcusdt_usdm_1h_features.csv"),
]
RESULT_FILES = {
    "walk_forward_metrics": Path("artifacts/metrics/futures_walk_forward_metrics.csv"),
    "long_short_backtest": Path("artifacts/metrics/futures_long_short_summary.csv"),
    "monthly_monitoring": Path("artifacts/metrics/futures_monitoring_monthly.csv"),
    "feature_drift": Path("artifacts/metrics/futures_feature_drift.csv"),
    "final_test_predictions": Path("artifacts/metrics/futures_final_test_predictions.csv"),
    "final_test_model_metrics": Path("artifacts/metrics/futures_final_test_model_metrics.csv"),
    "final_test_strategy_grid": Path("artifacts/metrics/futures_final_test_strategy_grid.csv"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL,
            phase TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            input_fingerprint TEXT NOT NULL,
            description TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES experiment_runs(run_id),
            role TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES experiment_runs(run_id),
            source_file TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            context_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_run_metrics_run
            ON run_metrics(run_id, source_file, metric_name);
        """
    )


def main() -> None:
    missing = [str(path) for path in [*INPUT_FILES, *RESULT_FILES.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("Run the Futures research steps first; missing: " + ", ".join(missing))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = {str(path): sha256_file(path) for path in INPUT_FILES}
    fingerprint = hashlib.sha256(
        json.dumps(input_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as connection:
        create_tables(connection)
        cursor = connection.execute(
            """INSERT INTO experiment_runs
               (run_name, phase, created_at_utc, input_fingerprint, description)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "binance_usdm_long_short_research",
                "phases_10_11_final_holdout",
                created_at,
                fingerprint,
                "Futures-native walk-forward model, long/short cost and funding scenarios, and drift monitoring; final test reserved.",
            ),
        )
        run_id = cursor.lastrowid

        artifacts = [
            ("input", path) for path in INPUT_FILES
        ] + [("result", path) for path in RESULT_FILES.values()]
        for role, path in artifacts:
            connection.execute(
                """INSERT INTO run_artifacts
                   (run_id, role, relative_path, sha256, size_bytes)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    run_id,
                    role,
                    path.as_posix(),
                    sha256_file(path),
                    path.stat().st_size,
                ),
            )

        metric_rows = []
        for source_name, path in RESULT_FILES.items():
            frame = pd.read_csv(path)
            for row_number, row in frame.iterrows():
                context = {}
                for key, value in row.items():
                    if pd.isna(value):
                        continue
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_rows.append(
                            (run_id, source_name, int(row_number), str(key), float(value), "")
                        )
                    else:
                        context[str(key)] = value.item() if hasattr(value, "item") else value
                # Attach row identifiers/settings to each numeric metric in that source row.
                if metric_rows:
                    for index in range(len(metric_rows) - 1, -1, -1):
                        metric = metric_rows[index]
                        if metric[0] != run_id or metric[1] != source_name or metric[2] != row_number:
                            break
                        metric_rows[index] = (*metric[:5], json.dumps(context, sort_keys=True, default=str))

        connection.executemany(
            """INSERT INTO run_metrics
               (run_id, source_file, row_number, metric_name, metric_value, context_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            metric_rows,
        )

    print(f"Registered research run: {run_id}")
    print(f"Input fingerprint: {fingerprint}")
    print(f"Artifacts tracked: {len(artifacts)}")
    print(f"Numeric metric records: {len(metric_rows)}")
    print(f"Saved SQLite registry to: {DB_PATH}")


if __name__ == "__main__":
    main()
