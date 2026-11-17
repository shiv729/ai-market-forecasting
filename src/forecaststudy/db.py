from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCHEMA_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (
    study_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK(mode IN ('real', 'synthetic')),
    deadline TEXT NOT NULL,
    manifest_version INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS markets (
    study_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT,
    event_group TEXT,
    question TEXT NOT NULL,
    resolution_rules TEXT NOT NULL,
    yes_asset_id TEXT NOT NULL,
    no_asset_id TEXT NOT NULL,
    yes_outcome_index INTEGER NOT NULL,
    no_outcome_index INTEGER NOT NULL,
    url TEXT NOT NULL,
    category TEXT,
    enrolled_at TEXT NOT NULL,
    expected_end_at TEXT NOT NULL,
    selection_json TEXT NOT NULL,
    raw_market_json TEXT NOT NULL,
    PRIMARY KEY (study_id, market_id),
    FOREIGN KEY (study_id) REFERENCES studies(study_id)
);
CREATE TABLE IF NOT EXISTS collection_runs (
    run_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    FOREIGN KEY (study_id) REFERENCES studies(study_id)
);
CREATE TABLE IF NOT EXISTS collection_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    study_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    scheduled_date TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES collection_runs(run_id),
    FOREIGN KEY (study_id, market_id) REFERENCES markets(study_id, market_id)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    study_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    scheduled_date TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    market_probability REAL,
    quote_method TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    spread REAL,
    quote_at TEXT,
    quote_status TEXT NOT NULL,
    quote_issue TEXT,
    raw_quote_json TEXT NOT NULL,
    UNIQUE (study_id, market_id, scheduled_date),
    FOREIGN KEY (run_id) REFERENCES collection_runs(run_id),
    FOREIGN KEY (study_id, market_id) REFERENCES markets(study_id, market_id)
);
CREATE TABLE IF NOT EXISTS forecast_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    completed_at TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    prompt_text TEXT,
    research_config_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    probability REAL,
    rationale TEXT,
    evidence_urls_json TEXT,
    uncertainties_json TEXT,
    raw_response TEXT,
    validation_status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd REAL,
    failure_reason TEXT,
    research_mode TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES collection_runs(run_id),
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS forecasts (
    snapshot_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE,
    forecast_at TEXT NOT NULL,
    forecast_probability REAL NOT NULL CHECK(forecast_probability >= 0 AND forecast_probability <= 1),
    rationale TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    time_gap_seconds REAL,
    gap_valid INTEGER NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id),
    FOREIGN KEY (attempt_id) REFERENCES forecast_attempts(attempt_id)
);
CREATE TABLE IF NOT EXISTS resolutions (
    resolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    outcome INTEGER CHECK(outcome IS NULL OR outcome IN (0, 1)),
    checked_at TEXT NOT NULL,
    resolved_at TEXT,
    source TEXT NOT NULL,
    review_required INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE (study_id, market_id, revision),
    FOREIGN KEY (study_id, market_id) REFERENCES markets(study_id, market_id)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_study_date ON snapshots(study_id, scheduled_date);
CREATE INDEX IF NOT EXISTS idx_forecast_attempts_snapshot ON forecast_attempts(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_resolutions_market ON resolutions(study_id, market_id, revision);
"""


def connect(path: str) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(SCHEMA)
    row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        connection.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row["version"] != SCHEMA_VERSION:
        raise RuntimeError(
            "Database schema {} is unsupported; this program supports {}".format(
                row["version"], SCHEMA_VERSION
            )
        )
    connection.commit()
    return connection


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def latest_resolution(
    connection: sqlite3.Connection, study_id: str, market_id: str
) -> Optional[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM resolutions WHERE study_id=? AND market_id=? "
        "ORDER BY revision DESC LIMIT 1",
        (study_id, market_id),
    ).fetchone()


def insert_resolution(
    connection: sqlite3.Connection,
    study_id: str,
    market_id: str,
    status: str,
    outcome: Optional[int],
    checked_at: str,
    resolved_at: Optional[str],
    source: str,
    review_required: bool,
    raw: Dict[str, Any],
) -> None:
    current = latest_resolution(connection, study_id, market_id)
    revision = 1 if current is None else int(current["revision"]) + 1
    connection.execute(
        "INSERT INTO resolutions(study_id,market_id,revision,status,outcome,checked_at,"
        "resolved_at,source,review_required,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            study_id,
            market_id,
            revision,
            status,
            outcome,
            checked_at,
            resolved_at,
            source,
            int(review_required),
            json_dumps(raw),
        ),
    )
