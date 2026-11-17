from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from forecaststudy.analysis import brier_score, evaluate
from forecaststudy.collection import collect
from forecaststudy.config import StudyConfig
from forecaststudy.db import connect, insert_resolution
from forecaststudy.demo import SyntheticPolymarketClient, run_demo
from forecaststudy.models import Forecast, iso_utc
from forecaststudy.polymarket import (
    PolymarketClient,
    market_to_candidate,
    quote_from_order_book,
    resolution_from_api,
)
from forecaststudy.providers import (
    ForecastOutputError,
    OpenAIResponsesProvider,
    SyntheticResearchProvider,
    TavilyResearchProvider,
    evidence_is_excluded,
)
from forecaststudy.study import canonical_hash, register_manifest


def test_brier_score_and_probability_validation() -> None:
    assert brier_score(0.65, 1) == pytest.approx(0.1225)
    assert brier_score(0.20, 0) == pytest.approx(0.04)
    for probability in (-0.1, 1.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            Forecast.validate(
                {
                    "probability": probability,
                    "rationale": "reason",
                    "evidence_urls": [],
                    "key_uncertainties": [],
                }
            )
    with pytest.raises(ValueError):
        Forecast.validate({"probability": 0.5})


def test_prediction_market_evidence_is_excluded() -> None:
    assert evidence_is_excluded(
    "https://metaculus.com/questions/123/example", "Unrelated content"
    )
    assert evidence_is_excluded(
    "https://news.example/story",
    "The prediction market gives this event a 70% chance.",
    )
    assert evidence_is_excluded(
    "https://polymarket.com/event/example", "Official resolution rules"
    )
    assert not evidence_is_excluded(
    "https://official.example/report", "The agency published the final result."
    )


def test_yes_no_token_mapping_uses_labels_not_array_order() -> None:
    raw = {
        "id": "17",
        "conditionId": "cond-17",
        "question": "Will the event occur?",
        "description": "Resolution rules are detailed and accessible.",
        "resolutionSource": "Official source",
        "active": True,
        "closed": False,
        "version": "v1",
        "outcomes": '["No", "Yes"]',
        "clobTokenIds": '["token-no", "token-yes"]',
        "endDateIso": "2027-01-01T00:00:00Z",
        "liquidityNum": 2000,
        "volumeNum": 5000,
        "events": [{"id": "ev1", "slug": "event"}],
        "slug": "market",
    }
    candidate, reasons = market_to_candidate(
        raw, "2027-12-31T23:59:59Z", 1000, 0
    )
    assert reasons == []
    assert candidate is not None
    assert candidate.yes_asset_id == "token-yes"
    assert candidate.no_asset_id == "token-no"
    assert candidate.yes_outcome_index == 1


def test_candidate_end_date_deadline_and_protocol_are_enforced() -> None:
    raw = {
        "id": "1",
        "question": "Q",
        "description": "Resolution rules.",
        "active": True,
        "closed": False,
        "version": "v3",
        "outcomes": '["Yes","No"]',
        "endDateIso": "2028-01-01T00:00:00Z",
        "liquidityNum": 1000,
    }
    candidate, reasons = market_to_candidate(raw, "2027-12-31T23:59:59Z", 1000, 0)
    assert candidate is None
    assert any("unsupported" in reason for reason in reasons)
    assert any("study deadline" in reason for reason in reasons)


def test_quote_requires_fresh_two_sided_valid_book() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh_book = {
        "timestamp": str(int(now.timestamp() * 1000)),
        "bids": [{"price": "0.4"}],
        "asks": [{"price": "0.5"}],
    }
    quote = quote_from_order_book(fresh_book, now, 60, 0.2)
    assert quote.status == "valid"
    assert quote.midpoint == pytest.approx(0.45)

    one_sided = quote_from_order_book(
        dict(fresh_book, asks=[]), now, 60, 0.2
    )
    assert one_sided.midpoint is None
    stale = dict(fresh_book, timestamp=str(int((now - timedelta(hours=1)).timestamp() * 1000)))
    assert quote_from_order_book(stale, now, 60, 0.2).status == "stale"
    wide = dict(fresh_book, asks=[{"price": "0.9"}])
    assert quote_from_order_book(wide, now, 60, 0.2).midpoint is None


def test_resolution_states_and_nonbinary_payouts_are_not_forced() -> None:
    checked = "2026-01-01T00:00:00Z"
    disputed = resolution_from_api(
        [{"condition_id": "c", "status": "challenged"}], checked, 0, 1
    )
    assert disputed["status"] == "disputed"
    assert disputed["outcome"] is None
    binary = resolution_from_api(
        [{"condition_id": "c", "status": "resolved", "payouts": [0, 1_000_000]}],
        checked,
        0,
        1,
    )
    assert binary["outcome"] == 0
    split = resolution_from_api(
        [{"condition_id": "c", "status": "resolved", "payouts": [500_000, 500_000]}],
        checked,
        0,
        1,
    )
    assert split["status"] == "ambiguous"
    assert split["outcome"] is None


def _insert_observation(
    connection: sqlite3.Connection,
    study_id: str,
    market_id: str,
    outcome: int,
    day: int,
    ai_probability: float,
    market_probability: float,
) -> None:
    snapshot_id = "{}-{}".format(market_id, day)
    attempt_id = "attempt-" + snapshot_id
    timestamp = "2026-02-{:02d}T12:00:00Z".format(day)
    connection.execute(
        "INSERT INTO snapshots(snapshot_id,run_id,study_id,market_id,scheduled_date,snapshot_at,"
        "market_probability,quote_method,best_bid,best_ask,spread,quote_at,quote_status,raw_quote_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            snapshot_id,
            "run",
            study_id,
            market_id,
            timestamp[:10],
            timestamp,
            market_probability,
            "YES best-bid/best-ask midpoint",
            market_probability - 0.01,
            market_probability + 0.01,
            0.02,
            timestamp,
            "valid",
            "{}",
        ),
    )
    connection.execute(
        "INSERT INTO forecast_attempts(attempt_id,run_id,snapshot_id,attempted_at,completed_at,"
        "provider,model,prompt_version,prompt_sha256,research_config_json,evidence_json,probability,"
        "rationale,evidence_urls_json,uncertainties_json,raw_response,validation_status,research_mode) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            attempt_id,
            "run",
            snapshot_id,
            timestamp,
            timestamp,
            "test",
            "test-model",
            "v1",
            "hash",
            "{}",
            "[]",
            ai_probability,
            "rationale",
            "[]",
            "[]",
            "{}",
            "valid",
            "manual",
        ),
    )
    connection.execute(
        "INSERT INTO forecasts(snapshot_id,attempt_id,forecast_at,forecast_probability,rationale,"
        "model,prompt_version,time_gap_seconds,gap_valid) VALUES(?,?,?,?,?,?,?,?,?)",
        (snapshot_id, attempt_id, timestamp, ai_probability, "rationale", "test-model", "v1", 0, 1),
    )


def test_equal_market_weighting_and_reproducible_bootstrap(tmp_path: Path) -> None:
    database = str(tmp_path / "study.sqlite")
    connection = connect(database)
    now = "2026-02-01T00:00:00Z"
    markets: List[Dict[str, Any]] = []
    for market_id, event_id in [("A", "event-A"), ("B", "event-B")]:
        markets.append(
            {
                "market_id": market_id,
                "event_id": event_id,
                "event_group": event_id,
                "question": "Q " + market_id,
                "resolution_rules": "Rules",
                "yes_asset_id": "yes-" + market_id,
                "no_asset_id": "no-" + market_id,
                "yes_outcome_index": 0,
                "no_outcome_index": 1,
                "url": "https://example.test/" + market_id,
                "expected_end_at": "2026-04-01T00:00:00Z",
                "raw": {"conditionId": "c-" + market_id},
                "selection_metadata": {},
            }
        )
    manifest: Dict[str, Any] = {
        "study_id": "weighted",
        "mode": "real",
        "manifest_version": 1,
        "created_at": now,
        "study_deadline": "2026-05-01T00:00:00Z",
        "study_config": {},
        "selection_criteria": {},
        "selection": {},
        "markets": markets,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    config = StudyConfig(study_id="weighted", mode="real")
    register_manifest(connection, manifest, config)
    connection.execute(
        "INSERT INTO collection_runs(run_id,study_id,started_at,mode,status,config_json) "
        "VALUES('run','weighted',?,'real','completed','{}')",
        (now,),
    )
    _insert_observation(connection, "weighted", "A", 1, 1, 0, 0.5)
    _insert_observation(connection, "weighted", "A", 1, 2, 0, 0.5)
    _insert_observation(connection, "weighted", "B", 0, 3, 0, 1)
    insert_resolution(connection, "weighted", "A", "resolved", 1, now, now, "test", False, {})
    insert_resolution(connection, "weighted", "B", "resolved", 0, now, now, "test", False, {})
    connection.commit()
    connection.close()

    first, _ = evaluate(database, "weighted", 500, 123, config_mode="real")
    second, _ = evaluate(database, "weighted", 500, 123, config_mode="real")
    assert first["delta"] == pytest.approx(-0.125)
    assert first["valid_paired_observations"] == 3
    assert first["bootstrap_ci_95"] == second["bootstrap_ci_95"]


def test_snapshot_uniqueness_prevents_duplicate_daily_observations(tmp_path: Path) -> None:
    connection = connect(str(tmp_path / "unique.sqlite"))
    connection.execute(
        "INSERT INTO studies(study_id,mode,deadline,manifest_version,manifest_json,manifest_sha256,config_json,created_at) "
        "VALUES('s','real','2027-01-01T00:00:00Z',1,'{}','hash','{}','2026-01-01T00:00:00Z')"
    )
    connection.execute(
        "INSERT INTO collection_runs(run_id,study_id,started_at,mode,status,config_json) "
        "VALUES('run','s','2026-01-01T00:00:00Z','real','completed','{}')"
    )
    connection.execute(
        "INSERT INTO markets(study_id,market_id,question,resolution_rules,yes_asset_id,no_asset_id,"
        "yes_outcome_index,no_outcome_index,url,enrolled_at,expected_end_at,selection_json,raw_market_json) "
        "VALUES('s','m','q','r','y','n',0,1,'url','2026-01-01T00:00:00Z','2026-12-01T00:00:00Z','{}','{}')"
    )
    values = ("id1", "s", "m", "2026-01-01", "2026-01-01T00:00:00Z", "midpoint", "valid", "{}")
    connection.execute(
        "INSERT INTO snapshots(snapshot_id,run_id,study_id,market_id,scheduled_date,snapshot_at,quote_method,quote_status,raw_quote_json) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        ("id1", "run") + values[1:],
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO snapshots(snapshot_id,run_id,study_id,market_id,scheduled_date,snapshot_at,quote_method,quote_status,raw_quote_json) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            ("id2", "run") + values[1:],
        )
    connection.close()


def test_public_polymarket_api_calls_need_no_auth_and_use_documented_paths() -> None:
    requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/book":
            return httpx.Response(
                200,
                json={
                    "timestamp": "1782753357257",
                    "bids": [{"price": "0.4", "size": "10"}],
                    "asks": [{"price": "0.5", "size": "10"}],
                },
            )
        return httpx.Response(200, json={"data": []})

    client = PolymarketClient(transport=httpx.MockTransport(handler))
    client.get_order_book("asset")
    client.get_resolution_state("condition")
    assert requests[0].url.host == "clob.polymarket.com"
    assert requests[0].url.path == "/book"
    assert requests[0].headers.get("authorization") is None
    assert requests[1].url.host == "data-api.polymarket.com"
    assert requests[1].url.path == "/v2/resolutions"
    client.close()


def test_tavily_evidence_is_timestamp_filtered_and_market_sources_excluded() -> None:
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://news.example/story",
                        "title": "Current report",
                        "content": "A public statement was released.",
                        "published_date": "2026-01-30T00:00:00Z",
                    },
                    {
                        "url": "https://news.example/future",
                        "content": "This has not yet happened.",
                        "published_date": "2026-02-02T00:00:00Z",
                    },
                    {
                        "url": "https://polymarket.com/event/example",
                        "content": "YES trades at 75%.",
                        "published_date": "2026-01-30T00:00:00Z",
                    },
                ]
            },
        )

    from forecaststudy.models import MarketCandidate

    provider = TavilyResearchProvider(
        "test-key", {"max_results": 5}, transport=httpx.MockTransport(handler)
    )
    market = MarketCandidate(
        market_id="m",
        event_id=None,
        question="Will the event occur?",
        resolution_rules="Official statements determine the outcome.",
        yes_asset_id="yes",
        no_asset_id="no",
        yes_outcome_index=0,
        no_outcome_index=1,
        url="https://example.invalid",
        category=None,
        event_group=None,
        expected_end_at="2026-03-01T00:00:00Z",
        liquidity=0,
        volume=0,
    )
    evidence = provider.retrieve(market, now, now)
    assert len(evidence) == 1
    assert evidence[0].url == "https://news.example/story"
    assert "75%" not in evidence[0].content
    assert requests[0].url == "https://api.tavily.com/search"
    payload = json.loads(requests[0].content)
    assert "market probability" not in payload["query"].lower()


def test_openai_responses_api_structured_output_is_validated_and_usage_saved() -> None:
    requests: List[httpx.Request] = []
    model_output = json.dumps(
        {
            "probability": 0.65,
            "rationale": "A concise evidence-based rationale.",
            "evidence_urls": ["https://news.example/story"],
            "key_uncertainties": ["The report may be incomplete."],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": model_output}],
                    }
                ],
                "usage": {"input_tokens": 123, "output_tokens": 34},
            },
        )

    provider = OpenAIResponsesProvider(
        "test-key", "gpt-4o-mini", transport=httpx.MockTransport(handler)
    )
    forecast, raw, input_tokens, output_tokens = provider.forecast("prompt", 200)
    assert forecast.probability == pytest.approx(0.65)
    assert input_tokens == 123 and output_tokens == 34
    assert requests[0].url == "https://api.openai.com/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert json.loads(raw)["status"] == "completed"

    malformed = OpenAIResponsesProvider(
        "test-key",
        "gpt-4o-mini",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {"content": [{"type": "output_text", "text": '{"probability": 4}'}]}
                    ],
                },
            )
        ),
    )
    with pytest.raises(ForecastOutputError) as error:
        malformed.forecast("prompt", 200)
    assert error.value.raw_response


def test_failed_forecast_is_not_scored_and_demo_reuses_daily_snapshot(tmp_path: Path) -> None:
    result = run_demo(str(tmp_path / "demo"), bootstrap_replicates=100)
    summary = result["summary"]
    assert summary["mode"] == "synthetic"
    assert summary["not_enough_resolved_data"] is False
    assert summary["valid_paired_observations"] == 12
    assert "SYNTHETIC" in Path(result["exports"]["report"]).read_text(encoding="utf-8")

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    config = StudyConfig(**manifest["study_config"])
    now = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00")).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    outcomes = {"demo-1": 1, "demo-2": 0, "demo-3": 1, "demo-4": 0, "demo-5": 1, "demo-6": 0}
    yes_indices = {"demo-1": 0, "demo-2": 1, "demo-3": 0, "demo-4": 1, "demo-5": 0, "demo-6": 1}
    api = SyntheticPolymarketClient(outcomes, yes_indices)
    api.now = now
    repeated = collect(
        result["manifest"],
        config,
        database_path=result["database"],
        market_api=api,
        now=now,
    )
    assert repeated["skipped"] == 6
    with sqlite3.connect(result["database"]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 12


def test_collection_failure_does_not_create_probability(tmp_path: Path) -> None:
    result = run_demo(str(tmp_path / "base"), bootstrap_replicates=100)
    base = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    failed = dict(base)
    failed["study_id"] = "synthetic-failure"
    failed["markets"] = base["markets"][:1]
    failed["study_config"] = dict(base["study_config"], study_id="synthetic-failure")
    failed.pop("content_sha256")
    failed["content_sha256"] = canonical_hash(failed)
    manifest_path = tmp_path / "failure-manifest.json"
    manifest_path.write_text(json.dumps(failed), encoding="utf-8")
    config = StudyConfig(**failed["study_config"])
    market = failed["markets"][0]
    now = datetime.fromisoformat(failed["created_at"].replace("Z", "+00:00")).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    api = SyntheticPolymarketClient({"demo-1": 1}, {"demo-1": 0})
    api.now = now

    class FailedForecast:
        name = "test"
        model = "bad-model"

        def forecast(self, user_prompt: str, max_output_tokens: int) -> Any:
            raise ValueError("malformed response")

    database_path = str(tmp_path / "failure.sqlite")
    dry_run = collect(
        str(manifest_path),
        config,
        database_path=database_path,
        market_api=api,
        research_provider=SyntheticResearchProvider(),
        forecast_provider=FailedForecast(),
        now=now,
        dry_run=True,
    )
    assert dry_run["model_calls"] == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0

    collection = collect(
        str(manifest_path),
        config,
        database_path=database_path,
        market_api=api,
        research_provider=SyntheticResearchProvider(),
        forecast_provider=FailedForecast(),
        now=now,
    )
    assert collection["failed"] == 2  # one malformed attempt and one bounded retry
    summary, _ = evaluate(
        database_path,
        "synthetic-failure",
        bootstrap_replicates=100,
        config_mode="synthetic",
    )
    assert summary["valid_paired_observations"] == 0
    assert summary["not_enough_resolved_data"] is True
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM forecast_attempts WHERE probability IS NULL"
        ).fetchone()[0] == 2
