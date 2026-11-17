from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .db import connect


def brier_score(probability: float, outcome: int) -> float:
    if outcome not in (0, 1):
        raise ValueError("binary outcome must be 0 or 1")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    return float((probability - outcome) ** 2)


def _observations(connection: sqlite3.Connection, study_id: str) -> pd.DataFrame:
    rows = connection.execute(
        """
        SELECT s.study_id, s.market_id, m.event_group, m.expected_end_at, m.url,
               s.scheduled_date, s.snapshot_at, s.market_probability, s.best_bid,
               s.best_ask, s.spread, s.quote_at, s.quote_status, s.quote_issue,
               f.forecast_at, f.forecast_probability, f.time_gap_seconds, f.gap_valid,
               r.outcome, r.status AS resolution_status, r.resolved_at,
               a.research_mode, a.provider, a.model, a.prompt_version
        FROM snapshots s
        JOIN markets m ON m.study_id=s.study_id AND m.market_id=s.market_id
        LEFT JOIN forecasts f ON f.snapshot_id=s.snapshot_id
        LEFT JOIN forecast_attempts a ON a.attempt_id=f.attempt_id
        LEFT JOIN (
            SELECT study_id, market_id, outcome, status, resolved_at,
                   ROW_NUMBER() OVER (PARTITION BY study_id,market_id ORDER BY revision DESC) AS rank
            FROM resolutions
        ) r ON r.study_id=s.study_id AND r.market_id=s.market_id AND r.rank=1
        WHERE s.study_id=?
        ORDER BY s.scheduled_date, s.market_id
        """,
        (study_id,),
    ).fetchall()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return frame
    for column in ["snapshot_at", "forecast_at", "expected_end_at", "resolved_at"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["market_brier"] = np.where(
        frame["market_probability"].notna() & frame["outcome"].isin([0, 1]),
        (frame["market_probability"] - frame["outcome"]) ** 2,
        np.nan,
    )
    frame["ai_brier"] = np.where(
        frame["forecast_probability"].notna() & frame["outcome"].isin([0, 1]),
        (frame["forecast_probability"] - frame["outcome"]) ** 2,
        np.nan,
    )
    frame["paired_valid"] = (
        frame["outcome"].isin([0, 1])
        & frame["market_probability"].notna()
        & frame["forecast_probability"].notna()
        & frame["gap_valid"].map(lambda value: bool(value) if pd.notna(value) else False)
        & (frame["quote_status"] == "valid")
    )
    frame["delta"] = frame["ai_brier"] - frame["market_brier"]
    frame["horizon_days"] = (
        (frame["expected_end_at"] - frame["snapshot_at"]).dt.total_seconds() / 86400
    )
    return frame


def _summarize_pairs(pairs: pd.DataFrame) -> Dict[str, Any]:
    if pairs.empty:
        return {
            "markets": 0,
            "observations": 0,
            "ai_brier": None,
            "market_brier": None,
            "delta": None,
            "baseline_brier": None,
            "market_wins": 0,
            "ties": 0,
            "market_wins_ai": 0,
        }
    per_market = pairs.groupby("market_id", as_index=False).agg(
        ai_brier=("ai_brier", "mean"),
        market_brier=("market_brier", "mean"),
        baseline_brier=("baseline_brier", "mean"),
    )
    per_market["delta"] = per_market.ai_brier - per_market.market_brier
    return {
        "markets": int(len(per_market)),
        "observations": int(len(pairs)),
        "ai_brier": float(per_market.ai_brier.mean()),
        "market_brier": float(per_market.market_brier.mean()),
        "delta": float(per_market.delta.mean()),
        "baseline_brier": float(per_market.baseline_brier.mean()),
        "market_wins": int((per_market.delta > 1e-12).sum()),
        "ties": int((per_market.delta.abs() <= 1e-12).sum()),
        "market_wins_ai": int((per_market.delta < -1e-12).sum()),
    }


def _bootstrap_ci(
    pairs: pd.DataFrame,
    replicates: int,
    seed: int,
    event_clusters: bool = False,
) -> Dict[str, Any]:
    if pairs.empty:
        return {"low": None, "high": None, "replicates": 0, "seed": seed}
    market_means = pairs.groupby("market_id", as_index=False).agg(
        ai_brier=("ai_brier", "mean"),
        market_brier=("market_brier", "mean"),
    )
    market_means["delta"] = market_means.ai_brier - market_means.market_brier
    if event_clusters:
        groups = (
            pairs[["market_id", "event_group"]]
            .drop_duplicates()
            .fillna({"event_group": ""})
            .set_index("market_id")["event_group"]
            .to_dict()
        )
        market_means["cluster"] = market_means.market_id.map(
            lambda market_id: groups.get(market_id) or market_id
        )
        cluster_ids = market_means.cluster.unique().tolist()
    else:
        market_means["cluster"] = market_means.market_id
        cluster_ids = market_means.market_id.tolist()
    if len(cluster_ids) < 2:
        return {"low": None, "high": None, "replicates": replicates, "seed": seed}
    rng = np.random.default_rng(seed)
    samples: List[float] = []
    by_cluster = {
        cluster: market_means[market_means.cluster == cluster].delta.to_numpy()
        for cluster in cluster_ids
    }
    for _ in range(replicates):
        draw = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        values = np.concatenate([by_cluster[item] for item in draw])
        samples.append(float(values.mean()))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "replicates": replicates,
        "seed": seed,
        "cluster_unit": "event" if event_clusters else "market",
    }


def _sensitivity(pairs: pd.DataFrame) -> Dict[str, Any]:
    if pairs.empty:
        return {"first": _summarize_pairs(pairs), "last": _summarize_pairs(pairs), "horizons": {}}
    ordered = pairs.sort_values(["market_id", "snapshot_at"])
    first = ordered.groupby("market_id", as_index=False).head(1)
    last = ordered.groupby("market_id", as_index=False).tail(1)
    buckets = {
        "<1 day": pairs[pairs.horizon_days < 1],
        "1-7 days": pairs[(pairs.horizon_days >= 1) & (pairs.horizon_days < 7)],
        "7-30 days": pairs[(pairs.horizon_days >= 7) & (pairs.horizon_days < 30)],
        "30+ days": pairs[pairs.horizon_days >= 30],
    }
    horizon_summary = {}
    for name, subset in buckets.items():
        market_count = int(subset.market_id.nunique())
        horizon_summary[name] = (
            _summarize_pairs(subset)
            if market_count >= 5
            else {"markets": market_count, "observations": int(len(subset)), "not_reported": "fewer than 5 markets"}
        )
    return {
        "first": _summarize_pairs(first),
        "last": _summarize_pairs(last),
        "horizons": horizon_summary,
    }


def _calibration(pairs: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    if pairs.empty:
        return []
    bins = np.linspace(0, 1, 11)
    labels = range(10)
    subset = pairs.copy()
    subset["bin"] = pd.cut(
        subset[column], bins=bins, labels=labels, include_lowest=True, right=True
    )
    result: List[Dict[str, Any]] = []
    for index in labels:
        bucket = subset[subset.bin == index]
        if bucket.empty:
            continue
        result.append(
            {
                "bin_start": float(index / 10),
                "bin_end": float((index + 1) / 10),
                "mean_forecast": float(bucket[column].mean()),
                "observed_frequency": float(bucket.outcome.mean()),
                "n": int(len(bucket)),
                "markets": int(bucket.market_id.nunique()),
            }
        )
    return result


def evaluate(
    database_path: str,
    study_id: str,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 729,
    event_clusters: bool = False,
    config_mode: Optional[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    connection = connect(database_path)
    study = connection.execute(
        "SELECT mode,manifest_json FROM studies WHERE study_id=?", (study_id,)
    ).fetchone()
    if study is None:
        connection.close()
        raise ValueError("Study not found: {}".format(study_id))
    mode = study["mode"]
    if config_mode is not None and config_mode != mode:
        connection.close()
        raise ValueError("Requested data mode does not match stored study mode")
    manifest = json.loads(study["manifest_json"])
    frame = _observations(connection, study_id)
    enrolled = int(
        connection.execute("SELECT COUNT(*) AS n FROM markets WHERE study_id=?", (study_id,)).fetchone()["n"]
    )
    forecasted = int(
        connection.execute(
            "SELECT COUNT(DISTINCT s.market_id) AS n FROM snapshots s JOIN forecasts f "
            "ON f.snapshot_id=s.snapshot_id WHERE s.study_id=?",
            (study_id,),
        ).fetchone()["n"]
    )
    attempts = connection.execute(
        "SELECT validation_status,COUNT(*) AS n FROM forecast_attempts "
        "WHERE run_id IN (SELECT run_id FROM collection_runs WHERE study_id=?) "
        "GROUP BY validation_status",
        (study_id,),
    ).fetchall()
    status_counts = {row["validation_status"]: int(row["n"]) for row in attempts}
    collection_events = connection.execute(
        "SELECT event_type,reason,COUNT(*) AS n FROM collection_events "
        "WHERE study_id=? GROUP BY event_type,reason ORDER BY event_type,reason",
        (study_id,),
    ).fetchall()
    collection_event_counts: Dict[str, int] = {}
    collection_event_reasons: List[Dict[str, Any]] = []
    for row in collection_events:
        collection_event_counts[row["event_type"]] = (
            collection_event_counts.get(row["event_type"], 0) + int(row["n"])
        )
        collection_event_reasons.append(
            {"event_type": row["event_type"], "reason": row["reason"], "count": int(row["n"])}
        )
    resolution_rows = connection.execute(
        "SELECT r.status,r.outcome FROM resolutions r JOIN (SELECT market_id,MAX(revision) revision "
        "FROM resolutions WHERE study_id=? GROUP BY market_id) x "
        "ON x.market_id=r.market_id AND x.revision=r.revision WHERE r.study_id=?",
        (study_id, study_id),
    ).fetchall()
    resolved_count = sum(1 for row in resolution_rows if row["outcome"] in (0, 1))
    unresolved_count = enrolled - resolved_count
    if frame.empty:
        pairs = frame
        pairs["baseline_brier"] = []
    else:
        pairs = frame[frame.paired_valid].copy()
        pairs["baseline_brier"] = (0.5 - pairs["outcome"]) ** 2
    summary = _summarize_pairs(pairs)
    ci = _bootstrap_ci(pairs, bootstrap_replicates, bootstrap_seed, event_clusters)
    market_delta = (
        pairs.groupby("market_id", as_index=False)
        .agg(ai_brier=("ai_brier", "mean"), market_brier=("market_brier", "mean"))
        if not pairs.empty
        else pd.DataFrame(columns=["market_id", "ai_brier", "market_brier"])
    )
    if not market_delta.empty:
        market_delta["delta"] = market_delta.ai_brier - market_delta.market_brier
    skill = None
    if summary["market_brier"] not in (None, 0):
        skill = (summary["market_brier"] - summary["ai_brier"]) / summary["market_brier"]
    quote_invalid = int((frame.quote_status != "valid").sum()) if not frame.empty else 0
    gap_invalid = (
        int((~frame.gap_valid.map(lambda value: bool(value) if pd.notna(value) else False)).sum())
        if not frame.empty
        else 0
    )
    latest = _latest_resolution_map(connection, study_id)
    summary.update(
        {
            "study_id": study_id,
            "experiment_arm": manifest.get("study_config", {}).get(
                "experiment_arm", "primary"
            ),
            "mode": mode,
            "manifest_sha256": manifest.get("content_sha256"),
            "study_deadline": manifest.get("study_deadline"),
            "configuration": manifest.get("study_config", {}),
            "enrolled_markets": enrolled,
            "forecasted_markets": forecasted,
            "resolved_markets": resolved_count,
            "unresolved_markets": unresolved_count,
            "valid_paired_observations": int(len(pairs)),
            "quote_exclusions": quote_invalid,
            "time_gap_exclusions": gap_invalid,
            "forecast_attempts": status_counts,
            "collection_events": collection_event_counts,
            "missingness_reasons": collection_event_reasons,
            "bootstrap_ci_95": ci,
            "skill_vs_market": skill,
            "sensitivity": _sensitivity(pairs),
            "calibration_ai": _calibration(pairs, "forecast_probability") if not pairs.empty else [],
            "calibration_market": _calibration(pairs, "market_probability") if not pairs.empty else [],
            "not_enough_resolved_data": len(pairs) == 0,
            "market_resolution_status": latest,
            "caveats": [
                "Daily forecasts within a market are averaged before markets receive equal weight.",
                "The bootstrap resamples markets (or event clusters), not daily rows.",
                "Related markets and shared evidence may remain statistically dependent.",
                "A sample of about 50 markets cannot establish universal forecasting superiority.",
                "Lower Brier score is not evidence of trading profitability.",
            ],
        }
    )
    if not frame.empty:
        observations = frame.copy()
    else:
        observations = pd.DataFrame()
    connection.close()
    return summary, observations


def _latest_resolution_map(connection: sqlite3.Connection, study_id: str) -> Dict[str, str]:
    rows = connection.execute(
        "SELECT r.market_id,r.status FROM resolutions r JOIN (SELECT market_id,MAX(revision) revision "
        "FROM resolutions WHERE study_id=? GROUP BY market_id) x "
        "ON x.market_id=r.market_id AND x.revision=r.revision WHERE r.study_id=?",
        (study_id, study_id),
    ).fetchall()
    return {str(row["market_id"]): str(row["status"]) for row in rows}
