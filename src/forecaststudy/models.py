from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import math


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def parse_json_array(value: Any) -> List[Any]:
    import json

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


@dataclass
class MarketCandidate:
    market_id: str
    event_id: Optional[str]
    question: str
    resolution_rules: str
    yes_asset_id: str
    no_asset_id: str
    yes_outcome_index: int
    no_outcome_index: int
    url: str
    category: Optional[str]
    event_group: Optional[str]
    expected_end_at: str
    liquidity: float
    volume: float
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Quote:
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    midpoint: Optional[float]
    quote_at: Optional[str]
    status: str
    issue: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    url: str
    retrieved_at: str
    published_at: Optional[str]
    title: str
    content: str
    provider: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Forecast:
    probability: float
    rationale: str
    evidence_urls: List[str]
    key_uncertainties: List[str]

    @classmethod
    def validate(cls, value: Any) -> "Forecast":
        if not isinstance(value, dict):
            raise ValueError("forecast output must be a JSON object")
        probability = value.get("probability")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("probability must be a finite number in [0, 1]")
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be a finite number in [0, 1]")
        rationale = value.get("rationale")
        urls = value.get("evidence_urls")
        uncertainties = value.get("key_uncertainties")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if not isinstance(urls, list) or not all(isinstance(item, str) for item in urls):
            raise ValueError("evidence_urls must be a list of strings")
        if not isinstance(uncertainties, list) or not all(
            isinstance(item, str) for item in uncertainties
        ):
            raise ValueError("key_uncertainties must be a list of strings")
        return cls(probability, rationale.strip(), urls, uncertainties)


@dataclass
class Resolution:
    status: str
    outcome: Optional[int]
    checked_at: str
    resolved_at: Optional[str]
    source: str
    raw: Dict[str, Any]
    review_required: bool = False


@dataclass
class StudyConfig:
    study_id: str = "ai-vs-markets-primary"
    mode: str = "real"
    study_deadline: Optional[str] = None
    forecast_cutoff_hours_before_end: int = 0
    min_liquidity_usd: float = 1000.0
    min_volume_usd: float = 0.0
    max_spread: float = 0.10
    max_quote_age_seconds: int = 300
    max_snapshot_forecast_gap_seconds: int = 300
    max_markets_per_run: int = 10
    max_research_calls_per_run: int = 10
    max_model_calls_per_run: int = 10
    max_output_tokens: int = 700
    max_retries: int = 2
    max_daily_spend_usd: Optional[float] = None
    estimated_input_usd_per_million_tokens: Optional[float] = None
    estimated_output_usd_per_million_tokens: Optional[float] = None
    research_provider: str = "tavily"
    evidence_file: str = "examples/evidence.json"
    model_provider: str = "openai"
    model: str = "gpt-4o-mini"
    prompt_version: str = "primary-v1"
    research_config: Dict[str, Any] = field(default_factory=dict)
    bootstrap_replicates: int = 5000
    bootstrap_seed: int = 729
    random_seed: int = 729
    manifest_version: int = 1
    data_dir: str = "data"

    def validate(self) -> None:
        if self.mode not in {"real", "synthetic"}:
            raise ValueError("mode must be 'real' or 'synthetic'")
        if not self.study_deadline:
            raise ValueError(
                "Set a study_deadline in config/study.json or STUDY_DEADLINE in .env"
            )
        if parse_datetime(self.study_deadline) is None:
            raise ValueError("study_deadline must be an ISO-8601 timestamp with timezone")
        if self.max_spread < 0 or self.max_spread > 1:
            raise ValueError("max_spread must be between 0 and 1")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.max_markets_per_run < 1:
            raise ValueError("max_markets_per_run must be at least one")
