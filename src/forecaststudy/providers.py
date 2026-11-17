from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlparse

import httpx

from .models import Evidence, Forecast, MarketCandidate, iso_utc, parse_datetime


FORECAST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability": {"type": "number"},
        "rationale": {"type": "string"},
        "evidence_urls": {"type": "array", "items": {"type": "string"}},
        "key_uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["probability", "rationale", "evidence_urls", "key_uncertainties"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / "primary-v1.txt"
).read_text(encoding="utf-8").strip()


def build_forecast_prompt(
    market: MarketCandidate,
    evidence: List[Evidence],
    current_time: str,
    cutoff: str,
) -> Tuple[str, str]:
    evidence_block = [
        {
            "url": item.url,
            "retrieved_at": item.retrieved_at,
            "published_at": item.published_at,
            "title": item.title,
            "content": item.content,
        }
        for item in evidence
    ]
    payload = {
        "question": market.question,
        "yes_condition": market.question,
        "resolution_rules": market.resolution_rules,
        "current_time_utc": current_time,
        "forecasting_cutoff_utc": cutoff,
        "evidence": evidence_block,
        "instruction": (
            "Make a forecast from information available by current_time_utc. "
            "If evidence is absent or uncertain, state this plainly."
        ),
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    prompt_hash = hashlib.sha256(
        (SYSTEM_PROMPT + "\n" + user_prompt).encode("utf-8")
    ).hexdigest()
    return user_prompt, prompt_hash


def evidence_is_excluded(url: str, content: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(".invalid") or host == "invalid":
        return True
    market_domains = (
        "polymarket.com",
        "polymarket.us",
        "metaculus.com",
        "manifold.markets",
        "manifold.money",
        "kalshi.com",
        "predictit.org",
    )
    if any(host == domain or host.endswith("." + domain) for domain in market_domains):
        return True
    text = (url + " " + content).lower()
    has_market_context = bool(
        re.search(
            r"\b(prediction market|forecast market|market-implied|market odds|polymarket|metaculus|manifold|kalshi|predictit)\b",
            text,
        )
    )
    has_price_language = bool(
        re.search(
            r"\b(odds|implied probability|probability|chance|trading at)\b|\b\d+(?:\.\d+)?\s*%",
            text,
        )
    )
    return has_market_context and has_price_language


def _within_cutoff(published_at: Optional[str], cutoff: datetime) -> bool:
    if not published_at:
        return True
    parsed = parse_datetime(published_at)
    return parsed is not None and parsed <= cutoff


class ResearchProvider(Protocol):
    name: str

    def retrieve(
        self, market: MarketCandidate, retrieved_at: datetime, cutoff: datetime
    ) -> List[Evidence]:
        ...


class ForecastProvider(Protocol):
    name: str
    model: str

    def forecast(
        self, user_prompt: str, max_output_tokens: int
    ) -> Tuple[Forecast, str, Optional[int], Optional[int]]:
        ...


class ForecastOutputError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class TavilyResearchProvider:
    name = "tavily"

    def __init__(
        self,
        api_key: Optional[str],
        config: Dict[str, Any],
        timeout: float = 20,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required for the Tavily research provider")
        self.api_key = api_key
        self.config = config
        self.client = httpx.Client(timeout=timeout, transport=transport)

    def retrieve(
        self, market: MarketCandidate, retrieved_at: datetime, cutoff: datetime
    ) -> List[Evidence]:
        # Search only the event and resolution language; no market price is supplied.
        query = "{} {}".format(market.question, market.resolution_rules[:500])
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.config.get("search_depth", "basic"),
            "max_results": min(20, max(1, int(self.config.get("max_results", 5)))),
            "topic": self.config.get("topic", "news"),
            "include_published_date": True,
            "include_answer": False,
        }
        response = self.client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Tavily response did not contain a results array")
        items: List[Evidence] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "")
            content = str(result.get("content") or "")
            title = str(result.get("title") or "")
            published = result.get("published_date")
            published_at = str(published) if published else None
            if (
                not url
                or not content
                or evidence_is_excluded(url, title + " " + content)
                or not _within_cutoff(published_at, cutoff)
            ):
                continue
            items.append(
                Evidence(
                    url=url,
                    retrieved_at=iso_utc(retrieved_at),
                    published_at=published_at,
                    title=title,
                    content=content,
                    provider=self.name,
                    raw=result,
                )
            )
        return items


class ManualEvidenceProvider:
    name = "manual"

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def retrieve(
        self, market: MarketCandidate, retrieved_at: datetime, cutoff: datetime
    ) -> List[Evidence]:
        if isinstance(self.data, dict):
            rows = self.data.get(market.market_id, self.data.get("default", []))
        else:
            rows = self.data
        if not isinstance(rows, list):
            raise ValueError("Manual evidence must be a list or a market-ID keyed object")
        result: List[Evidence] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "")
            content = str(row.get("content") or row.get("excerpt") or "")
            title = str(row.get("title") or "")
            published = str(row["published_at"]) if row.get("published_at") else None
            if not url or not content:
                continue
            if evidence_is_excluded(url, title + " " + content) or not _within_cutoff(published, cutoff):
                continue
            result.append(
                Evidence(
                    url=url,
                    retrieved_at=iso_utc(retrieved_at),
                    published_at=published,
                    title=title,
                    content=content,
                    provider=self.name,
                    raw=row,
                )
            )
        return result


class NoEvidenceResearchProvider:
    name = "none"

    def retrieve(
        self, market: MarketCandidate, retrieved_at: datetime, cutoff: datetime
    ) -> List[Evidence]:
        return []


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        timeout: float = 45,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for real model forecasts")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=timeout, transport=transport)

    @staticmethod
    def _output_text(response: Dict[str, Any]) -> str:
        # The Responses API returns output items; extract only model text parts.
        fragments: List[str] = []
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
        if fragments:
            return "\n".join(fragments)
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        raise ValueError("OpenAI response has no output text (possibly a refusal or incomplete response)")

    def forecast(
        self, user_prompt: str, max_output_tokens: int
    ) -> Tuple[Forecast, str, Optional[int], Optional[int]]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "binary_forecast",
                    "strict": True,
                    "schema": FORECAST_SCHEMA,
                }
            },
        }
        response = self.client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": "Bearer " + self.api_key},
            json=payload,
        )
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("status") != "completed":
            raise ForecastOutputError(
                "OpenAI response did not complete: {}".format(response_data.get("status")),
                json.dumps(response_data, ensure_ascii=False),
            )
        try:
            raw_text = self._output_text(response_data)
        except ValueError as exc:
            raise ForecastOutputError(
                str(exc), json.dumps(response_data, ensure_ascii=False)
            ) from exc
        try:
            forecast = Forecast.validate(json.loads(raw_text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ForecastOutputError(str(exc), json.dumps(response_data, ensure_ascii=False)) from exc
        usage = response_data.get("usage") if isinstance(response_data.get("usage"), dict) else {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        return (
            forecast,
            json.dumps(response_data, ensure_ascii=False),
            int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
            int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
        )


class SyntheticResearchProvider:
    name = "synthetic"

    def retrieve(
        self, market: MarketCandidate, retrieved_at: datetime, cutoff: datetime
    ) -> List[Evidence]:
        content = (
            "SYNTHETIC DEMO ONLY: A fabricated public update says the event is "
            "plausible but uncertain. This is not real evidence."
        )
        return [
            Evidence(
                url="https://example.invalid/synthetic/evidence",
                retrieved_at=iso_utc(retrieved_at),
                published_at=iso_utc(retrieved_at),
                title="Synthetic evidence fixture",
                content=content,
                provider=self.name,
                raw={"synthetic": True},
            )
        ]


class SyntheticForecastProvider:
    name = "synthetic"
    model = "synthetic-deterministic-v1"

    def forecast(
        self, user_prompt: str, max_output_tokens: int
    ) -> Tuple[Forecast, str, Optional[int], Optional[int]]:
        digest = hashlib.sha256(user_prompt.encode("utf-8")).digest()
        probability = 0.25 + (int.from_bytes(digest[:2], "big") / 65535) * 0.5
        result = Forecast(
            probability=round(probability, 4),
            rationale="SYNTHETIC DEMO ONLY: deterministic fixture forecast; not a model judgment.",
            evidence_urls=["https://example.invalid/synthetic/evidence"],
            key_uncertainties=["All evidence and forecasts in this run are fabricated."],
        )
        return result, json.dumps(result.__dict__), None, None
