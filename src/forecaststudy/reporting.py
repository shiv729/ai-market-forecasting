from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .analysis import evaluate


def markdown_report(summary: Dict[str, Any]) -> str:
    if summary.get("not_enough_resolved_data"):
        result = "**Not enough resolved data for a paired Brier-score comparison.**"
    else:
        ci = summary.get("bootstrap_ci_95", {})
        interval = (
            " [{:.4f}, {:.4f}]".format(ci["low"], ci["high"])
            if ci.get("low") is not None
            else " (CI unavailable: fewer than two independent bootstrap units)"
        )
        result = (
            "- AI Brier: **{:.4f}**\n"
            "- Market Brier: **{:.4f}**\n"
            "- Delta (AI − market): **{:.4f}**{}\n"
            "- 0.5 baseline Brier: **{:.4f}**".format(
                summary["ai_brier"],
                summary["market_brier"],
                summary["delta"],
                interval,
                summary["baseline_brier"],
            )
        )
    statuses = summary.get("market_resolution_status", {})
    lines = [
        "# Can AI Forecast Better Than Prediction Markets?",
        "",
        "> Data mode: **{}**. This report is generated from the saved study database.".format(
            summary.get("mode", "unknown").upper()
        ),
        "",
        "## Study coverage",
        "",
        "- Study ID: `{}`".format(summary.get("study_id")),
        "- Experiment arm: `{}`".format(summary.get("experiment_arm", "primary")),
        "- Manifest SHA-256: `{}`".format(summary.get("manifest_sha256")),
        "- Study deadline: `{}`".format(summary.get("study_deadline")),
        "- Configuration: `{}`".format(
            json.dumps(summary.get("configuration", {}), sort_keys=True)
        ),
        "- Enrolled / forecasted / resolved / unresolved markets: **{} / {} / {} / {}**".format(
            summary.get("enrolled_markets", 0),
            summary.get("forecasted_markets", 0),
            summary.get("resolved_markets", 0),
            summary.get("unresolved_markets", 0),
        ),
        "- Valid paired observations: **{}**".format(summary.get("valid_paired_observations", 0)),
        "- Quote / timestamp-gap exclusions: **{} / {}**".format(
            summary.get("quote_exclusions", 0), summary.get("time_gap_exclusions", 0)
        ),
        "- Attempt outcomes: `{}`".format(json.dumps(summary.get("forecast_attempts", {}), sort_keys=True)),
        "",
        "## Primary result",
        "",
        result,
        "",
        "Negative delta means the AI forecast had a lower Brier score. Scores average daily observations within each market, then weight each market equally.",
        "",
        "## Market-level paired results",
        "",
        "- AI better / tie / market better: **{} / {} / {}**".format(
            summary.get("market_wins_ai", 0),
            summary.get("ties", 0),
            summary.get("market_wins", 0),
        ),
        "- Skill relative to market: `{}`".format(summary.get("skill_vs_market")),
        "- Resolution status counts: `{}`".format(json.dumps(statuses, sort_keys=True)),
        "",
        "## Interpretation limits",
        "",
    ]
    lines.extend("- " + item for item in summary.get("caveats", []))
    lines += [
        "- Source filtering reduces, but cannot eliminate, forecast leakage from public reporting.",
        "- Scheduled market end dates are not guaranteed resolution dates. Disputed, ambiguous, canceled, and non-binary outcomes are retained but are not scored as binary outcomes.",
        "",
        "## Sensitivity analyses",
        "",
        "- First valid paired forecast per market: `{}`".format(
            json.dumps(summary.get("sensitivity", {}).get("first", {}), sort_keys=True)
        ),
        "- Last valid paired forecast per market: `{}`".format(
            json.dumps(summary.get("sensitivity", {}).get("last", {}), sort_keys=True)
        ),
        "- Forecast horizon buckets: `{}`".format(
            json.dumps(summary.get("sensitivity", {}).get("horizons", {}), sort_keys=True)
        ),
        "",
        "## Reproducibility",
        "",
        "- Bootstrap resampling unit: `{}`; replicates: {}; seed: {}.".format(
            summary.get("bootstrap_ci_95", {}).get("cluster_unit", "market"),
            summary.get("bootstrap_ci_95", {}).get("replicates", 0),
            summary.get("bootstrap_ci_95", {}).get("seed"),
        ),
        "- Synthetic-mode outputs are illustrative fixtures and are not real forecasts or evidence.",
        "",
    ]
    return "\n".join(lines)


def export_study(
    database_path: str,
    study_id: str,
    output_dir: str,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 729,
    config_mode: str = "real",
) -> Dict[str, str]:
    summary, observations = evaluate(
        database_path,
        study_id,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        config_mode=config_mode,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    observations_path = target / "observations.csv"
    market_path = target / "market_differences.csv"
    report_path = target / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if observations.empty:
        pd.DataFrame().to_csv(observations_path, index=False)
        pd.DataFrame().to_csv(market_path, index=False)
    else:
        observations.insert(0, "experiment_arm", summary.get("experiment_arm", "primary"))
        observations.insert(0, "data_mode", config_mode)
        observations.to_csv(observations_path, index=False)
        scored = observations[observations.get("paired_valid", False)].copy()
        if not scored.empty:
            market_differences = scored.groupby("market_id", as_index=False).agg(
                ai_brier=("ai_brier", "mean"),
                market_brier=("market_brier", "mean"),
                observations=("market_id", "size"),
            ).assign(
                delta=lambda frame: frame["ai_brier"] - frame["market_brier"]
            )
            market_differences.insert(0, "data_mode", config_mode)
            market_differences.to_csv(market_path, index=False)
        else:
            pd.DataFrame().to_csv(market_path, index=False)
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "observations": str(observations_path),
        "market_differences": str(market_path),
        "report": str(report_path),
    }
