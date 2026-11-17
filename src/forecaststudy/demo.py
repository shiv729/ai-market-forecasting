from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from .collection import collect, resolve_study
from .config import StudyConfig
from .models import iso_utc
from .polymarket import resolution_from_api
from .reporting import export_study
from .study import canonical_hash


class SyntheticPolymarketClient:
    def __init__(self, outcomes: Dict[str, int], yes_indices: Dict[str, int]) -> None:
        self.outcomes = outcomes
        self.yes_indices = yes_indices
        self.now = datetime.now(timezone.utc)
        self.resolved = False

    def get_order_book(self, asset_id: str) -> Dict[str, Any]:
        marker = sum(ord(char) for char in asset_id) % 25
        midpoint = 0.34 + marker / 100 + (self.now.day % 5) / 100
        return {
            "asset_id": asset_id,
            "timestamp": str(int(self.now.timestamp() * 1000)),
            "bids": [{"price": "{:.3f}".format(midpoint - 0.02), "size": "500"}],
            "asks": [{"price": "{:.3f}".format(midpoint + 0.02), "size": "500"}],
            "hash": "synthetic",
        }

    def get_resolution_state(self, condition_id: str) -> List[Dict[str, Any]]:
        market_id = condition_id.replace("condition-", "")
        if market_id not in self.outcomes:
            return []
        if not self.resolved:
            return [{"condition_id": condition_id, "status": "initialized", "payouts": None}]
        yes_index = self.yes_indices[market_id]
        winning_index = yes_index if self.outcomes[market_id] == 1 else 1 - yes_index
        payouts = [1_000_000 if index == winning_index else 0 for index in range(2)]
        return [
            {
                "condition_id": condition_id,
                "status": "resolved",
                "payouts": payouts,
                "resolved_at": iso_utc(self.now),
                "reporter": "SYNTHETIC",
            }
        ]

    def get_market(self, market_id: str) -> Dict[str, Any]:
        return {"id": market_id, "active": True, "closed": False}


def run_demo(data_dir: str, bootstrap_replicates: int = 1000) -> Dict[str, Any]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = StudyConfig(
        study_id="synthetic-demo",
        mode="synthetic",
        study_deadline=iso_utc(datetime.now(timezone.utc) + timedelta(days=90)),
        forecast_cutoff_hours_before_end=0,
        min_liquidity_usd=0,
        min_volume_usd=0,
        max_spread=0.1,
        max_quote_age_seconds=300,
        max_snapshot_forecast_gap_seconds=300,
        max_markets_per_run=10,
        max_research_calls_per_run=10,
        max_model_calls_per_run=10,
        max_output_tokens=300,
        max_retries=1,
        research_provider="synthetic",
        model_provider="synthetic",
        model="synthetic-deterministic-v1",
        prompt_version="synthetic-demo-v1",
        research_config={"synthetic": True},
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=729,
        random_seed=729,
        data_dir=str(root),
    )
    outcomes = {"demo-1": 1, "demo-2": 0, "demo-3": 1, "demo-4": 0, "demo-5": 1, "demo-6": 0}
    yes_indices = {"demo-1": 0, "demo-2": 1, "demo-3": 0, "demo-4": 1, "demo-5": 0, "demo-6": 1}
    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    markets: List[Dict[str, Any]] = []
    for index, (market_id, outcome) in enumerate(outcomes.items()):
        yes_index = yes_indices[market_id]
        labels = ["YES", "NO"] if yes_index == 0 else ["NO", "YES"]
        assets = [market_id + "-yes", market_id + "-no"] if yes_index == 0 else [market_id + "-no", market_id + "-yes"]
        markets.append(
            {
                "market_id": market_id,
                "event_id": "event-" + market_id,
                "event_group": "event-" + market_id,
                "question": "Will the synthetic demo event {} occur?".format(index + 1),
                "resolution_rules": "Synthetic fixture only: resolves YES if its fabricated event occurs by the stated date.",
                "yes_asset_id": assets[yes_index],
                "no_asset_id": assets[1 - yes_index],
                "yes_outcome_index": yes_index,
                "no_outcome_index": 1 - yes_index,
                "url": "https://example.invalid/synthetic/{}".format(market_id),
                "category": "Synthetic",
                "expected_end_at": iso_utc(now + timedelta(days=30 + index)),
                "liquidity": 1000 + index * 100,
                "volume": 5000 + index * 100,
                "raw": {
                    "id": market_id,
                    "conditionId": "condition-" + market_id,
                    "outcomes": labels,
                    "active": True,
                    "closed": False,
                },
                "selection_metadata": {"synthetic": True, "fixture_outcome": outcome},
            }
        )
    payload: Dict[str, Any] = {
        "study_id": config.study_id,
        "mode": config.mode,
        "manifest_version": config.manifest_version,
        "created_at": iso_utc(now),
        "study_deadline": config.study_deadline,
        "study_config": config.__dict__,
        "selection_criteria": {"synthetic_fixture": True},
        "selection": {
            "requested_count": len(markets),
            "enrolled_count": len(markets),
            "random_selection": False,
            "random_seed": None,
            "event_group_deduplication": "synthetic fixtures are independent",
            "exclusions": [],
        },
        "markets": markets,
    }
    payload["content_sha256"] = canonical_hash(payload)
    manifest_path = root / "study-manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    api = SyntheticPolymarketClient(outcomes, yes_indices)
    database_path = str(root / "research.sqlite")
    for day in range(2):
        api.now = now + timedelta(days=day)
        collect(
            str(manifest_path),
            config,
            database_path=database_path,
            market_api=api,
            now=api.now,
        )
    api.now = now + timedelta(days=2)
    api.resolved = True
    resolution_counts = resolve_study(
        str(manifest_path), config, database_path=database_path, market_api=api
    )
    exports = export_study(
        database_path,
        config.study_id,
        str(root / "exports"),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=config.bootstrap_seed,
        config_mode="synthetic",
    )
    with open(exports["summary"], "r", encoding="utf-8") as file:
        summary = json.load(file)
    return {
        "mode": "synthetic",
        "study_id": config.study_id,
        "manifest": str(manifest_path),
        "database": database_path,
        "resolution_counts": resolution_counts,
        "exports": exports,
        "summary": summary,
    }
