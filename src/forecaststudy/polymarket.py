from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import httpx

from .models import MarketCandidate, Quote, iso_utc, parse_datetime, parse_json_array, utc_now


GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"


class PolymarketClient:
    """Read-only public Polymarket API client."""

    def __init__(
        self,
        timeout_seconds: float = 15,
        max_retries: int = 2,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self.client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(method, url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = min(float(retry_after), 10.0) if retry_after else 0.5 * (2**attempt)
                        except ValueError:
                            delay = 0.5 * (2**attempt)
                        time.sleep(max(0.0, delay))
                        continue
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt == self.max_retries:
                    raise
                time.sleep(0.5 * (2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Polymarket request failed without a response")

    def list_markets(
        self,
        deadline: str,
        min_liquidity: float,
        min_volume: float,
        max_pages: int = 10,
    ) -> List[Dict[str, Any]]:
        cursor: Optional[str] = None
        markets: List[Dict[str, Any]] = []
        for _ in range(max_pages):
            params: Dict[str, Any] = {
                "closed": "false",
                "limit": 100,
                "end_date_max": deadline,
                "liquidity_num_min": min_liquidity,
                "volume_num_min": min_volume,
                "include_tag": "true",
            }
            if cursor:
                params["after_cursor"] = cursor
            page = self._request("GET", GAMMA_BASE + "/markets/keyset", params=params)
            if isinstance(page, list):
                markets.extend(page)
                break
            if not isinstance(page, dict):
                raise ValueError("Gamma markets response was not a JSON object")
            rows = page.get("markets", [])
            if not isinstance(rows, list):
                raise ValueError("Gamma markets response has no markets array")
            markets.extend(item for item in rows if isinstance(item, dict))
            cursor = page.get("next_cursor")
            if not cursor or not rows:
                break
        return markets

    def get_order_book(self, asset_id: str) -> Dict[str, Any]:
        value = self._request(
            "GET", CLOB_BASE + "/book", params={"token_id": asset_id}
        )
        if not isinstance(value, dict):
            raise ValueError("CLOB order-book response was not a JSON object")
        return value

    def get_market(self, market_id: str) -> Dict[str, Any]:
        value = self._request("GET", GAMMA_BASE + "/markets/" + quote(market_id, safe=""))
        if not isinstance(value, dict):
            raise ValueError("Gamma market response was not a JSON object")
        return value

    def get_resolution_state(self, condition_id: str) -> List[Dict[str, Any]]:
        value = self._request(
            "GET",
            DATA_BASE + "/v2/resolutions",
            params={"condition": condition_id},
        )
        if not isinstance(value, dict) or not isinstance(value.get("data"), list):
            raise ValueError("Resolution API response did not contain a data array")
        return [row for row in value["data"] if isinstance(row, dict)]


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def market_to_candidate(
    raw: Dict[str, Any],
    study_deadline: str,
    min_liquidity: float,
    min_volume: float,
) -> Tuple[Optional[MarketCandidate], List[str]]:
    reasons: List[str] = []
    market_id = str(raw.get("id") or "")
    question = _text(raw.get("question"))
    if not market_id:
        reasons.append("missing market ID")
    if not question:
        reasons.append("missing question")
    if raw.get("active") is not True:
        reasons.append("not active")
    if raw.get("closed") is not False:
        reasons.append("closed or closure status unknown")
    if raw.get("archived") is True:
        reasons.append("archived")

    events = raw.get("events") if isinstance(raw.get("events"), list) else []
    event = events[0] if events and isinstance(events[0], dict) else {}
    description = _text(raw.get("description")) or _text(event.get("description"))
    resolution_source = _text(raw.get("resolutionSource")) or _text(event.get("resolutionSource"))
    if not description:
        reasons.append("missing full accessible resolution rules")
    rule_parts = []
    if description:
        rule_parts.append(description)
    if resolution_source and resolution_source not in description:
        rule_parts.append("Resolution source: " + resolution_source)
    rules = "\n\n".join(rule_parts)

    labels = parse_json_array(raw.get("outcomes"))
    if len(labels) != 2 or not all(isinstance(label, str) for label in labels):
        reasons.append("not a two-outcome market")
    normalized = [str(label).strip().lower() for label in labels]
    if sorted(normalized) != ["no", "yes"]:
        reasons.append("outcomes are not labeled exactly YES and NO")

    version = _text(raw.get("version")).lower()
    if not _text(raw.get("conditionId")):
        reasons.append("missing condition ID required for resolution verification")
    if version == "v2":
        asset_ids = raw.get("positionIds")
        if not isinstance(asset_ids, list):
            asset_ids = []
    elif version == "v1":
        asset_ids = parse_json_array(raw.get("clobTokenIds"))
    else:
        asset_ids = []
        reasons.append("unsupported or missing market protocol version")
    if len(asset_ids) != 2 or not all(isinstance(item, (str, int)) for item in asset_ids):
        reasons.append("missing outcome asset identifiers")

    yes_index = normalized.index("yes") if "yes" in normalized else -1
    no_index = normalized.index("no") if "no" in normalized else -1
    yes_id = str(asset_ids[yes_index]) if len(asset_ids) == 2 and yes_index >= 0 else ""
    no_id = str(asset_ids[no_index]) if len(asset_ids) == 2 and no_index >= 0 else ""

    end_value = raw.get("endDateIso") or raw.get("endDate")
    end_at = parse_datetime(end_value) if isinstance(end_value, str) else None
    deadline = parse_datetime(study_deadline)
    if end_at is None:
        reasons.append("missing or invalid expected end date")
    elif deadline is not None and end_at >= deadline:
        reasons.append("expected end date is not before study deadline")

    liquidity = _number(raw.get("liquidityNum", raw.get("liquidity"))) or 0.0
    volume = _number(raw.get("volumeNum", raw.get("volume"))) or 0.0
    if liquidity < min_liquidity:
        reasons.append("liquidity below minimum")
    if volume < min_volume:
        reasons.append("volume below minimum")
    if reasons:
        return None, reasons

    event_slug = _text(event.get("slug"))
    url = (
        "https://polymarket.com/event/{}".format(quote(event_slug, safe=""))
        if event_slug
        else GAMMA_BASE + "/markets/" + quote(market_id, safe="")
    )
    event_id = str(event["id"]) if event.get("id") is not None else None
    event_group = str(event.get("slug") or event_id or "") or None
    category = _text(raw.get("category")) or _text(event.get("category")) or None
    return (
        MarketCandidate(
            market_id=market_id,
            event_id=event_id,
            question=question,
            resolution_rules=rules,
            yes_asset_id=yes_id,
            no_asset_id=no_id,
            yes_outcome_index=yes_index,
            no_outcome_index=no_index,
            url=url,
            category=category,
            event_group=event_group,
            expected_end_at=iso_utc(end_at),
            liquidity=liquidity,
            volume=volume,
            raw=raw,
        ),
        [],
    )


def _book_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        epoch = float(value)
        if epoch > 10_000_000_000:
            epoch /= 1000
        try:
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        parsed = parse_datetime(value)
        if parsed is not None:
            return parsed
        try:
            epoch = float(value)
        except ValueError:
            return None
        return _book_timestamp(epoch)
    return None


def quote_from_order_book(
    book: Dict[str, Any],
    received_at: datetime,
    max_age_seconds: int,
    max_spread: float,
) -> Quote:
    raw_bids = book.get("bids")
    raw_asks = book.get("asks")
    bids = [_number(item.get("price")) for item in raw_bids if isinstance(item, dict)] if isinstance(raw_bids, list) else []
    asks = [_number(item.get("price")) for item in raw_asks if isinstance(item, dict)] if isinstance(raw_asks, list) else []
    bids = [value for value in bids if value is not None]
    asks = [value for value in asks if value is not None]
    bid = max(bids) if bids else None
    ask = min(asks) if asks else None
    quote_at = _book_timestamp(book.get("timestamp"))
    if not bids or not asks:
        return Quote(bid, ask, None, None, iso_utc(quote_at) if quote_at else None, "invalid", "one-sided or empty order book", book)
    if bid is None or ask is None or not (0 <= bid <= 1 and 0 <= ask <= 1) or bid >= ask:
        return Quote(bid, ask, None, None, iso_utc(quote_at) if quote_at else None, "invalid", "invalid or crossed YES-token quote", book)
    spread = ask - bid
    midpoint = (bid + ask) / 2
    if quote_at is None:
        return Quote(bid, ask, spread, None, None, "invalid", "order-book timestamp is missing or invalid", book)
    age = (received_at.astimezone(timezone.utc) - quote_at).total_seconds()
    if age < -5:
        return Quote(bid, ask, spread, None, iso_utc(quote_at), "invalid", "order-book timestamp is in the future", book)
    if age > max_age_seconds:
        return Quote(bid, ask, spread, None, iso_utc(quote_at), "stale", "order-book quote exceeds maximum age", book)
    if spread > max_spread:
        return Quote(bid, ask, spread, None, iso_utc(quote_at), "too_wide", "spread exceeds configured maximum", book)
    return Quote(bid, ask, spread, midpoint, iso_utc(quote_at), "valid", None, book)


def resolution_from_api(
    rows: Iterable[Dict[str, Any]],
    checked_at: str,
    yes_outcome_index: int,
    no_outcome_index: int,
) -> Dict[str, Any]:
    row = next((item for item in rows if item.get("condition_id") or item.get("status")), None)
    if row is None:
        return {
            "status": "not_found",
            "outcome": None,
            "checked_at": checked_at,
            "resolved_at": None,
            "source": "Polymarket Data API v2 /v2/resolutions",
            "review_required": True,
            "raw": {"data": list(rows)},
        }
    state = str(row.get("status") or "unknown").lower()
    resolved_at = row.get("resolved_at") if isinstance(row.get("resolved_at"), str) else None
    payouts = row.get("payouts")
    status = state
    outcome: Optional[int] = None
    review = False
    if state == "resolved":
        if (
            isinstance(payouts, list)
            and len(payouts) == 2
            and yes_outcome_index in (0, 1)
            and no_outcome_index in (0, 1)
            and yes_outcome_index != no_outcome_index
        ):
            try:
                values = [int(value) for value in payouts]
            except (TypeError, ValueError):
                values = []
            if values == [1_000_000, 0]:
                outcome = 1 if yes_outcome_index == 0 else 0
            elif values == [0, 1_000_000]:
                outcome = 1 if yes_outcome_index == 1 else 0
            elif values == [0, 0] or values == [500_000, 500_000]:
                status = "canceled" if values == [0, 0] else "ambiguous"
                review = True
            else:
                status = "non_binary_settlement"
                review = True
        else:
            status = "ambiguous"
            review = True
    elif state in {"disputed", "challenged", "arbitration"}:
        status = "disputed"
        review = True
    elif state in {"proposed", "posed", "reproposed"}:
        status = "outcome_proposed"
        review = True
    elif state in {"initialized", "active"}:
        status = "pending"
    elif "cancel" in state:
        status = "canceled"
        review = True
    else:
        status = state or "unknown"
        review = True
    return {
        "status": status,
        "outcome": outcome,
        "checked_at": checked_at,
        "resolved_at": resolved_at,
        "source": "Polymarket Data API v2 /v2/resolutions",
        "review_required": review,
        "raw": row,
    }
