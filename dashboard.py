from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from forecaststudy.analysis import evaluate


st.set_page_config(page_title="AI vs Prediction Markets", layout="wide")
st.title("Can AI Forecast Better Than Prediction Markets?")
st.caption("Prospective forecast collection and Brier-score evaluation")

database_path = st.sidebar.text_input("SQLite database", os.getenv("STUDY_DATABASE", "data/research.sqlite"))
study_id = st.sidebar.text_input("Study ID", os.getenv("STUDY_ID", "ai-vs-markets-primary"))
bootstrap = st.sidebar.number_input("Bootstrap replicates", min_value=100, max_value=50000, value=5000, step=100)
seed = st.sidebar.number_input("Bootstrap seed", min_value=0, value=729, step=1)

if not Path(database_path).exists():
    st.info("No study database exists yet. Run the synthetic demo or enroll a real study.")
    st.stop()

try:
    result, observations = evaluate(database_path, study_id, int(bootstrap), int(seed))
except (sqlite3.Error, ValueError, OSError) as exc:
    st.error("Could not load this study: {}".format(exc))
    st.stop()

mode = result.get("mode", "unknown")
st.info("Data mode: **{}**{}".format(mode.upper(), " — synthetic fixtures are not real results." if mode == "synthetic" else ""))
st.subheader("Frozen study configuration")
st.json(result.get("configuration", {}))

columns = st.columns(5)
for column, label, value in zip(
    columns,
    ["Enrolled", "Forecasted", "Resolved", "Paired observations", "Unresolved"],
    [
        result.get("enrolled_markets", 0),
        result.get("forecasted_markets", 0),
        result.get("resolved_markets", 0),
        result.get("valid_paired_observations", 0),
        result.get("unresolved_markets", 0),
    ],
):
    column.metric(label, value)

if result.get("not_enough_resolved_data"):
    st.warning("Not enough resolved data for a paired Brier-score comparison. No results are imputed.")
else:
    columns = st.columns(4)
    ci = result.get("bootstrap_ci_95") or {}
    ci_text = (
        "{:.4f} to {:.4f}".format(ci["low"], ci["high"])
        if ci.get("low") is not None
        else "not estimable"
    )
    for column, label, value in zip(
        columns,
        ["AI Brier (lower is better)", "Market Brier", "AI − market", "95% market bootstrap CI"],
        [
            "{:.4f}".format(result["ai_brier"]),
            "{:.4f}".format(result["market_brier"]),
            "{:.4f}".format(result["delta"]),
            ci_text,
        ],
    ):
        column.metric(label, value)

if not observations.empty:
    st.subheader("Market probability trajectories")
    valid_markets = observations.market_id.dropna().unique().tolist()
    selected_market = st.selectbox("Market", valid_markets)
    trajectory = observations[observations.market_id == selected_market].sort_values("snapshot_at")
    line = trajectory.set_index("snapshot_at")[["market_probability", "forecast_probability"]].rename(
        columns={"market_probability": "Market", "forecast_probability": "AI"}
    )
    st.line_chart(line)
    st.caption("AI forecasts without an accepted snapshot remain missing rather than being filled.")

    if not result.get("not_enough_resolved_data"):
        st.subheader("Market-level paired Brier differences")
        scored = observations[observations.paired_valid].copy()
        if not scored.empty:
            market_differences = scored.groupby("market_id").agg(
                ai_brier=("ai_brier", "mean"),
                market_brier=("market_brier", "mean"),
            )
            market_differences["AI − market"] = market_differences.ai_brier - market_differences.market_brier
            st.bar_chart(market_differences[["AI − market"]])
            st.caption("Negative values favor AI; each market is weighted equally in the primary summary.")

            st.subheader("Calibration")
            st.caption("Calibration bins show sample counts; results are descriptive and often noisy at this sample size.")
            calibration = []
            for name, items in [("AI", result["calibration_ai"]), ("Market", result["calibration_market"])]:
                calibration.extend(dict(item, forecaster=name) for item in items)
            if calibration:
                st.dataframe(pd.DataFrame(calibration), use_container_width=True)
                figure, axis = plt.subplots(figsize=(6, 5))
                axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
                for name, items, color in [
                    ("AI", result["calibration_ai"], "tab:blue"),
                    ("Market", result["calibration_market"], "tab:orange"),
                ]:
                    if not items:
                        continue
                    x = [item["mean_forecast"] for item in items]
                    y = [item["observed_frequency"] for item in items]
                    axis.plot(x, y, marker="o", label=name, color=color)
                    for item in items:
                        axis.annotate(str(item["n"]), (item["mean_forecast"], item["observed_frequency"]), fontsize=8)
                axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean forecast probability", ylabel="Observed YES frequency")
                axis.legend()
                axis.grid(alpha=0.2)
                st.pyplot(figure)
                plt.close(figure)
else:
    st.info("No daily snapshots have been recorded.")

st.subheader("Exclusions, collection failures, and resolution review")
left, right = st.columns(2)
left.metric("Invalid / stale / wide quotes", result.get("quote_exclusions", 0))
right.metric("Snapshot–forecast gaps outside tolerance", result.get("time_gap_exclusions", 0))
st.dataframe(pd.DataFrame(result.get("missingness_reasons", [])), use_container_width=True)
st.json(result.get("forecast_attempts", {}))
st.dataframe(pd.DataFrame([{"market_id": key, "resolution_status": value} for key, value in result.get("market_resolution_status", {}).items()]), use_container_width=True)

try:
    with sqlite3.connect(database_path) as connection:
        markets = pd.read_sql_query(
            "SELECT market_id,question,url,category,expected_end_at FROM markets WHERE study_id=?",
            connection,
            params=(study_id,),
        )
    st.subheader("Enrolled markets")
    st.dataframe(markets, use_container_width=True, column_config={"url": st.column_config.LinkColumn("Market")})
except (sqlite3.Error, pd.errors.DatabaseError):
    st.warning("Market details are unavailable in this database.")

st.caption(
    "Leakage filtering cannot ensure independence. Related markets may be dependent. "
    "A lower Brier score does not establish trading profitability or universal superiority."
)
