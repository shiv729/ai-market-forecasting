from __future__ import annotations

import hashlib
import platform
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import StudyConfig
from .db import connect, insert_resolution, json_dumps, latest_resolution
from .models import (
    Evidence,
    MarketCandidate,
    Quote,
    iso_utc,
    parse_datetime,
    utc_now,
)
from .polymarket import PolymarketClient, quote_from_order_book, resolution_from_api
from .providers import (
    ManualEvidenceProvider,
    NoEvidenceResearchProvider,
    OpenAIResponsesProvider,
    SyntheticForecastProvider,
    SyntheticResearchProvider,
    TavilyResearchProvider,
    SYSTEM_PROMPT,
    build_forecast_prompt,
)
from .study import manifest_from_file, register_manifest


def _candidate(item: Dict[str, Any]) -> MarketCandidate:
    return MarketCandidate(
        market_id=str(item["market_id"]),
        event_id=item.get("event_id"),
        question=str(item["question"]),
        resolution_rules=str(item["resolution_rules"]),
        yes_asset_id=str(item["yes_asset_id"]),
        no_asset_id=str(item["no_asset_id"]),
        yes_outcome_index=int(item["yes_outcome_index"]),
        no_outcome_index=int(item["no_outcome_index"]),
        url=str(item.get("url") or ""),
        category=item.get("category"),
        event_group=item.get("event_group"),
        expected_end_at=str(item["expected_end_at"]),
        liquidity=float(item.get("liquidity") or 0),
        volume=float(item.get("volume") or 0),
        raw=item.get("raw") if isinstance(item.get("raw"), dict) else {},
    )


def _providers(config: StudyConfig) -> Tuple[Any, Any]:
    if config.mode == "synthetic":
        return SyntheticResearchProvider(), SyntheticForecastProvider()
    if config.research_provider == "tavily":
        research = TavilyResearchProvider(
            os.getenv("TAVILY_API_KEY"), config.research_config
        )
    elif config.research_provider == "none":
        research = NoEvidenceResearchProvider()
    elif config.research_provider == "manual":
        evidence_path = Path(config.evidence_file)
        if not evidence_path.exists():
            raise FileNotFoundError("Manual evidence file not found: {}".format(evidence_path))
        research = ManualEvidenceProvider(str(evidence_path))
    else:
        raise ValueError("Unsupported research provider: {}".format(config.research_provider))
    if config.model_provider != "openai":
        raise ValueError("Unsupported model provider: {}".format(config.model_provider))
    forecast = OpenAIResponsesProvider(
        os.getenv("OPENAI_API_KEY"), config.model
    )
    return research, forecast


def _market_state(
    api: Any, market: MarketCandidate, checked_at: Optional[str] = None
) -> Dict[str, Any]:
    rows = api.get_resolution_state(str(market.raw.get("conditionId") or ""))
    resolution = resolution_from_api(
        rows,
        checked_at or iso_utc(utc_now()),
        market.yes_outcome_index,
        market.no_outcome_index,
    )
    if resolution["status"] == "not_found":
        gamma = api.get_market(market.market_id)
        if gamma.get("active") is True and gamma.get("closed") is False:
            resolution["status"] = "pending"
            resolution["review_required"] = False
            resolution["raw"] = {"gamma_market": gamma, "resolution_rows": rows}
        else:
            resolution["status"] = "closed_unverified"
            resolution["review_required"] = True
            resolution["raw"] = {"gamma_market": gamma, "resolution_rows": rows}
    return resolution


def _market_cutoff(market: MarketCandidate, config: StudyConfig) -> datetime:
    end = parse_datetime(market.expected_end_at)
    if end is None:
        raise ValueError("Market {} has no valid expected end time".format(market.market_id))
    event_cutoff = end - timedelta(hours=config.forecast_cutoff_hours_before_end)
    study_deadline = parse_datetime(config.study_deadline)
    if study_deadline is not None:
        return min(event_cutoff, study_deadline)
    return event_cutoff


def _validate_frozen_config(manifest: Dict[str, Any], config: StudyConfig) -> None:
    if manifest.get("study_id") != config.study_id:
        raise ValueError("Configured study_id does not match the frozen manifest")
    frozen = manifest.get("study_config", {})
    for key in (
        "mode",
        "study_deadline",
        "experiment_arm",
        "forecast_cutoff_hours_before_end",
        "model_provider",
        "model",
        "prompt_version",
        "research_provider",
        "research_config",
        "max_output_tokens",
    ):
        if key in frozen and frozen[key] != getattr(config, key):
            raise ValueError(
                "Primary study setting '{}' differs from the frozen manifest; "
                "use a new study ID for a material configuration change".format(key)
            )


def _estimate_cost(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    config: StudyConfig,
) -> Optional[float]:
    if (
        config.estimated_input_usd_per_million_tokens is None
        or config.estimated_output_usd_per_million_tokens is None
        or input_tokens is None
        or output_tokens is None
    ):
        return None
    return (
        input_tokens * config.estimated_input_usd_per_million_tokens
        + output_tokens * config.estimated_output_usd_per_million_tokens
    ) / 1_000_000


def _daily_spend(connection: sqlite3.Connection, study_id: str, day: str) -> float:
    row = connection.execute(
        "SELECT COALESCE(SUM(estimated_cost_usd),0) AS total FROM forecast_attempts "
        "WHERE run_id IN (SELECT run_id FROM collection_runs WHERE study_id=?) "
        "AND substr(attempted_at,1,10)=?",
        (study_id, day),
    ).fetchone()
    return float(row["total"] or 0)


def _insert_attempt(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
    run_id: str,
    snapshot_id: str,
    attempted_at: str,
    provider: str,
    model: str,
    prompt_version: str,
    prompt_hash: str,
    research_config: Dict[str, Any],
    evidence: List[Evidence],
    prompt_text: Optional[str] = None,
    forecast: Any = None,
    raw_response: Optional[str] = None,
    validation_status: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    estimated_cost: Optional[float] = None,
    failure_reason: Optional[str] = None,
    research_mode: str,
    completed_at: Optional[str] = None,
) -> None:
    connection.execute(
        "INSERT INTO forecast_attempts(attempt_id,run_id,snapshot_id,attempted_at,completed_at,"
        "provider,model,prompt_version,prompt_sha256,research_config_json,evidence_json,probability,"
        "prompt_text,"
        "rationale,evidence_urls_json,uncertainties_json,raw_response,validation_status,input_tokens,"
        "output_tokens,estimated_cost_usd,failure_reason,research_mode) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            attempt_id,
            run_id,
            snapshot_id,
            attempted_at,
            completed_at,
            provider,
            model,
            prompt_version,
            prompt_hash,
            json_dumps(research_config),
            json_dumps([item.to_dict() for item in evidence]),
            forecast.probability if forecast is not None else None,
            prompt_text,
            forecast.rationale if forecast is not None else None,
            json_dumps(forecast.evidence_urls) if forecast is not None else None,
            json_dumps(forecast.key_uncertainties) if forecast is not None else None,
            raw_response,
            validation_status,
            input_tokens,
            output_tokens,
            estimated_cost,
            failure_reason,
            research_mode,
        ),
    )


def collect(
    manifest_path: str,
    config: StudyConfig,
    database_path: Optional[str] = None,
    market_api: Optional[Any] = None,
    research_provider: Optional[Any] = None,
    forecast_provider: Optional[Any] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    manifest = manifest_from_file(manifest_path)
    _validate_frozen_config(manifest, config)
    current_time = (now or utc_now()).astimezone(timezone.utc)
    clock = (lambda: current_time) if now is not None or config.mode == "synthetic" else utc_now
    database_path = database_path or str(Path(config.data_dir) / "research.sqlite")
    connection = connect(database_path)
    register_manifest(connection, manifest, config)
    if config.mode == "real":
        api = market_api or PolymarketClient(max_retries=config.max_retries)
    else:
        api = market_api
        if api is None:
            raise ValueError("Synthetic study collection requires a synthetic market adapter")
    if (research_provider is None or forecast_provider is None) and not dry_run:
        research, forecaster = _providers(config)
        research_provider = research_provider or research
        forecast_provider = forecast_provider or forecaster

    run_id = str(uuid.uuid4())
    import httpx
    import numpy
    import pandas
    from . import __version__

    run_config = {
        "study_config": config.__dict__,
        "runtime": {
            "forecaststudy": __version__,
            "python": platform.python_version(),
            "httpx": httpx.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        },
    }
    connection.execute(
        "INSERT INTO collection_runs(run_id,study_id,started_at,mode,status,config_json) "
        "VALUES(?,?,?,?,?,?)",
        (run_id, manifest["study_id"], iso_utc(current_time), config.mode, "running", json_dumps(run_config)),
    )
    connection.commit()
    counts = {"processed": 0, "forecasted": 0, "skipped": 0, "failed": 0, "model_calls": 0, "research_calls": 0}
    markets = manifest.get("markets", [])[: config.max_markets_per_run]
    model_calls = 0
    research_calls = 0
    errors: List[str] = []

    def log_event(market_id: str, event_type: str, reason: str, raw: Optional[Dict[str, Any]] = None) -> None:
        connection.execute(
            "INSERT INTO collection_events(event_id,run_id,study_id,market_id,scheduled_date,"
            "occurred_at,event_type,reason,raw_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                run_id,
                manifest["study_id"],
                market_id,
                current_time.date().isoformat(),
                iso_utc(clock()),
                event_type,
                reason,
                json_dumps(raw or {}),
            ),
        )

    try:
        for item in markets:
            market = _candidate(item)
            scheduled_date = current_time.date().isoformat()
            counts["processed"] += 1
            if current_time >= _market_cutoff(market, config):
                counts["skipped"] += 1
                log_event(market.market_id, "cutoff", "forecast cutoff reached")
                continue
            try:
                status = _market_state(api, market, iso_utc(clock()))
                insert_resolution(
                    connection,
                    manifest["study_id"],
                    market.market_id,
                    status["status"],
                    status["outcome"],
                    status["checked_at"],
                    status["resolved_at"],
                    status["source"],
                    status["review_required"],
                    status["raw"],
                )
                connection.commit()
            except Exception as exc:
                counts["failed"] += 1
                errors.append("{} resolution check: {}".format(market.market_id, type(exc).__name__))
                log_event(
                    market.market_id,
                    "resolution_check_failed",
                    "{}: {}".format(type(exc).__name__, exc),
                )
                connection.commit()
                continue
            if status["outcome"] is not None or status["review_required"]:
                counts["skipped"] += 1
                log_event(
                    market.market_id,
                    "not_forecasted",
                    "outcome status requires stopping/review: {}".format(status["status"]),
                    status["raw"],
                )
                connection.commit()
                continue

            existing = connection.execute(
                "SELECT s.snapshot_id FROM snapshots s WHERE s.study_id=? AND s.market_id=? "
                "AND s.scheduled_date=?",
                (manifest["study_id"], market.market_id, scheduled_date),
            ).fetchone()
            if existing is not None:
                if connection.execute(
                    "SELECT 1 FROM forecasts WHERE snapshot_id=?", (existing["snapshot_id"],)
                ).fetchone():
                    counts["skipped"] += 1
                    log_event(
                        market.market_id,
                        "already_collected",
                        "successful primary observation already exists for this UTC date",
                    )
                    connection.commit()
                    continue
                snapshot_id = existing["snapshot_id"]
                snapshot_row = connection.execute(
                    "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
                ).fetchone()
                snapshot_at = snapshot_row["snapshot_at"]
                market_probability = snapshot_row["market_probability"]
                if dry_run:
                    counts["skipped"] += 1
                    log_event(
                        market.market_id,
                        "dry_run",
                        "existing primary snapshot inspected; no research/model calls made",
                    )
                    connection.commit()
                    continue
            else:
                snapshot_at_dt = clock()
                snapshot_at = iso_utc(snapshot_at_dt)
                snapshot_id = str(uuid.uuid4())
                try:
                    book = api.get_order_book(market.yes_asset_id)
                    quote = quote_from_order_book(
                        book,
                        snapshot_at_dt,
                        config.max_quote_age_seconds,
                        config.max_spread,
                    )
                except Exception as exc:
                    quote = Quote(None, None, None, None, None, "invalid", "order-book request failed: {}".format(type(exc).__name__), {"error": str(exc)})
                market_probability = quote.midpoint
                if dry_run:
                    counts["skipped"] += 1
                    log_event(
                        market.market_id,
                        "dry_run",
                        "quote checked but not stored as the primary observation; no research/model calls made",
                        quote.raw,
                    )
                    connection.commit()
                    continue
                connection.execute(
                    "INSERT INTO snapshots(snapshot_id,run_id,study_id,market_id,scheduled_date,snapshot_at,"
                    "market_probability,quote_method,best_bid,best_ask,spread,quote_at,quote_status,"
                    "quote_issue,raw_quote_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        run_id,
                        manifest["study_id"],
                        market.market_id,
                        scheduled_date,
                        snapshot_at,
                        market_probability,
                        "YES best-bid/best-ask midpoint",
                        quote.best_bid,
                        quote.best_ask,
                        quote.spread,
                        quote.quote_at,
                        quote.status,
                        quote.issue,
                        json_dumps(quote.raw),
                    ),
                )
                connection.commit()

            if research_calls >= config.max_research_calls_per_run:
                counts["skipped"] += 1
                log_event(
                    market.market_id,
                    "research_limit",
                    "maximum research calls per run reached",
                )
                connection.commit()
                continue
            evidence: List[Evidence] = []
            research_succeeded = False
            research_error: Optional[Exception] = None
            for research_attempt in range(config.max_retries + 1):
                if research_calls >= config.max_research_calls_per_run:
                    break
                attempt_started = iso_utc(clock())
                research_calls += 1
                try:
                    research_cutoff = min(clock(), _market_cutoff(market, config))
                    evidence = research_provider.retrieve(
                        market, parse_datetime(attempt_started) or current_time, research_cutoff
                    )
                    research_succeeded = True
                    break
                except Exception as exc:
                    research_error = exc
                    attempt_id = str(uuid.uuid4())
                    _insert_attempt(
                        connection,
                        attempt_id=attempt_id,
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        attempted_at=attempt_started,
                        completed_at=iso_utc(clock()),
                        provider=str(getattr(forecast_provider, "name", "unknown")),
                        model=str(getattr(forecast_provider, "model", config.model)),
                        prompt_version=config.prompt_version,
                        prompt_hash="",
                        research_config=config.research_config,
                        evidence=evidence,
                        validation_status="research_failed",
                        failure_reason="{}: {}".format(type(exc).__name__, str(exc)),
                        research_mode=config.research_provider,
                    )
                    connection.commit()
                    counts["failed"] += 1
                    if research_attempt < config.max_retries:
                        time.sleep(min(0.5 * (2**research_attempt), 5))
            if not research_succeeded:
                reason = (
                    "maximum research calls per run reached"
                    if research_calls >= config.max_research_calls_per_run and research_error is None
                    else "{}: {}".format(type(research_error).__name__, research_error)
                )
                if research_error is not None:
                    errors.append("{} research failed".format(market.market_id))
                log_event(
                    market.market_id,
                    "research_limit" if research_error is None else "research_failed",
                    reason,
                )
                connection.commit()
                continue

            forecast_at: Optional[str] = None
            succeeded = False
            for _ in range(config.max_retries + 1):
                if model_calls >= config.max_model_calls_per_run:
                    break
                now_text = iso_utc(clock())
                prompt, prompt_hash = build_forecast_prompt(
                    market, evidence, now_text, iso_utc(_market_cutoff(market, config))
                )
                prompt_text = json_dumps(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                estimated_prompt_tokens = max(1, len(prompt) // 4)
                if config.max_daily_spend_usd is not None:
                    if (
                        config.estimated_input_usd_per_million_tokens is None
                        or config.estimated_output_usd_per_million_tokens is None
                    ):
                        failure = "daily spend cap requires both configured token price estimates"
                        attempt_id = str(uuid.uuid4())
                        _insert_attempt(
                            connection,
                            attempt_id=attempt_id,
                            run_id=run_id,
                            snapshot_id=snapshot_id,
                            attempted_at=iso_utc(clock()),
                            completed_at=iso_utc(clock()),
                            provider=forecast_provider.name,
                            model=forecast_provider.model,
                            prompt_version=config.prompt_version,
                            prompt_hash=prompt_hash,
                            prompt_text=prompt_text,
                            research_config=config.research_config,
                            evidence=evidence,
                            validation_status="cost_cap_unavailable",
                            failure_reason=failure,
                            research_mode=config.research_provider,
                        )
                        connection.commit()
                        counts["failed"] += 1
                        errors.append("{} cost cap requires price estimates".format(market.market_id))
                        log_event(
                            market.market_id,
                            "cost_cap_configuration",
                            failure,
                        )
                        connection.commit()
                        break
                    reserve = (
                        estimated_prompt_tokens * config.estimated_input_usd_per_million_tokens
                        + config.max_output_tokens * config.estimated_output_usd_per_million_tokens
                    ) / 1_000_000
                    spend = _daily_spend(connection, manifest["study_id"], current_time.date().isoformat())
                    if spend + reserve > config.max_daily_spend_usd:
                        attempt_id = str(uuid.uuid4())
                        _insert_attempt(
                            connection,
                            attempt_id=attempt_id,
                            run_id=run_id,
                            snapshot_id=snapshot_id,
                            attempted_at=iso_utc(clock()),
                            completed_at=iso_utc(clock()),
                            provider=forecast_provider.name,
                            model=forecast_provider.model,
                            prompt_version=config.prompt_version,
                            prompt_hash=prompt_hash,
                            prompt_text=prompt_text,
                            research_config=config.research_config,
                            evidence=evidence,
                            validation_status="cost_cap_reached",
                            failure_reason="estimated request would exceed configured daily spend cap",
                            research_mode=config.research_provider,
                        )
                        connection.commit()
                        counts["skipped"] += 1
                        log_event(
                            market.market_id,
                            "cost_cap",
                            "estimated request would exceed configured daily spend cap",
                        )
                        connection.commit()
                        break

                attempt_id = str(uuid.uuid4())
                attempted_at = iso_utc(clock())
                model_calls += 1
                try:
                    forecast, raw_response, input_tokens, output_tokens = forecast_provider.forecast(
                        prompt, config.max_output_tokens
                    )
                    available_urls = {item.url for item in evidence}
                    unsupported_urls = set(forecast.evidence_urls) - available_urls
                    if unsupported_urls:
                        raise ValueError(
                            "forecast cited URLs not present in the retrieved evidence"
                        )
                    forecast_at = iso_utc(clock())
                    cost = _estimate_cost(input_tokens, output_tokens, config)
                    _insert_attempt(
                        connection,
                        attempt_id=attempt_id,
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        attempted_at=attempted_at,
                        completed_at=forecast_at,
                        provider=forecast_provider.name,
                        model=forecast_provider.model,
                        prompt_version=config.prompt_version,
                        prompt_hash=prompt_hash,
                        prompt_text=prompt_text,
                        research_config=config.research_config,
                        evidence=evidence,
                        forecast=forecast,
                        raw_response=raw_response,
                        validation_status="valid",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        estimated_cost=cost,
                        research_mode=config.research_provider,
                    )
                    snapshot_dt = parse_datetime(snapshot_at)
                    gap = (
                        (parse_datetime(forecast_at) - snapshot_dt).total_seconds()
                        if snapshot_dt is not None
                        else None
                    )
                    gap_valid = gap is not None and 0 <= gap <= config.max_snapshot_forecast_gap_seconds
                    connection.execute(
                        "INSERT INTO forecasts(snapshot_id,attempt_id,forecast_at,forecast_probability,"
                        "rationale,model,prompt_version,time_gap_seconds,gap_valid) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            attempt_id,
                            forecast_at,
                            forecast.probability,
                            forecast.rationale,
                            forecast_provider.model,
                            config.prompt_version,
                            gap,
                            int(gap_valid),
                        ),
                    )
                    connection.commit()
                    succeeded = True
                    counts["forecasted"] += 1
                    break
                except Exception as exc:
                    completed = iso_utc(clock())
                    raw = getattr(exc, "raw_response", getattr(exc, "response_text", None))
                    _insert_attempt(
                        connection,
                        attempt_id=attempt_id,
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        attempted_at=attempted_at,
                        completed_at=completed,
                        provider=forecast_provider.name,
                        model=forecast_provider.model,
                        prompt_version=config.prompt_version,
                        prompt_hash=prompt_hash,
                        prompt_text=prompt_text,
                        research_config=config.research_config,
                        evidence=evidence,
                        raw_response=raw,
                        validation_status="invalid_or_failed",
                        failure_reason="{}: {}".format(type(exc).__name__, str(exc)),
                        research_mode=config.research_provider,
                    )
                    connection.commit()
                    counts["failed"] += 1
                    errors.append("{} model attempt failed ({})".format(market.market_id, type(exc).__name__))
                    if model_calls < config.max_model_calls_per_run and _ < config.max_retries:
                        time.sleep(min(0.5 * (2**_), 5))
            if not succeeded and model_calls >= config.max_model_calls_per_run:
                log_event(
                    market.market_id,
                    "model_limit",
                    "market was not forecast because model-call limit was reached",
                )
                connection.commit()
                counts["skipped"] += 1

        connection.execute(
            "UPDATE collection_runs SET completed_at=?,status=?,config_json=? WHERE run_id=?",
            (
                iso_utc(clock()),
                "partial" if errors else "completed",
                json_dumps(run_config),
                run_id,
            ),
        )
        connection.commit()
    finally:
        if config.mode == "real" and isinstance(api, PolymarketClient):
            api.close()
    counts["model_calls"] = model_calls
    counts["research_calls"] = research_calls
    counts["run_id"] = run_id
    counts["errors"] = errors
    connection.close()
    return counts


def resolve_study(
    manifest_path: str,
    config: StudyConfig,
    database_path: Optional[str] = None,
    market_api: Optional[Any] = None,
) -> Dict[str, int]:
    manifest = manifest_from_file(manifest_path)
    database_path = database_path or str(Path(config.data_dir) / "research.sqlite")
    connection = connect(database_path)
    register_manifest(connection, manifest, config)
    api = market_api or PolymarketClient(max_retries=config.max_retries)
    counts = {"resolved": 0, "pending": 0, "disputed": 0, "review": 0, "failed": 0}
    try:
        for item in manifest.get("markets", []):
            market = _candidate(item)
            try:
                state = _market_state(api, market)
                insert_resolution(
                    connection,
                    manifest["study_id"],
                    market.market_id,
                    state["status"],
                    state["outcome"],
                    state["checked_at"],
                    state["resolved_at"],
                    state["source"],
                    state["review_required"],
                    state["raw"],
                )
                connection.commit()
                if state["outcome"] in (0, 1):
                    counts["resolved"] += 1
                elif state["status"] == "disputed":
                    counts["disputed"] += 1
                elif state["status"] == "pending":
                    counts["pending"] += 1
                else:
                    counts["review"] += 1
            except Exception as exc:
                checked_at = iso_utc(utc_now())
                insert_resolution(
                    connection,
                    manifest["study_id"],
                    market.market_id,
                    "check_failed",
                    None,
                    checked_at,
                    None,
                    "Polymarket Data API v2 /v2/resolutions",
                    True,
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
                connection.commit()
                counts["failed"] += 1
    finally:
        if isinstance(api, PolymarketClient):
            api.close()
    connection.close()
    return counts
