from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import StudyConfig
from .db import connect, json_dumps
from .models import MarketCandidate, iso_utc, utc_now
from .polymarket import quote_from_order_book


def canonical_hash(value: Dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_candidate_report(
    rows: List[Dict[str, Any]],
    config: StudyConfig,
    client: Any,
    report_path: str,
    max_candidates: int = 500,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for raw in rows:
        candidate, reasons = MarketCandidateFromRaw(raw, config)
        if candidate is None:
            exclusions.append({"market_id": raw.get("id"), "question": raw.get("question"), "reasons": reasons, "raw": raw})
            continue
        try:
            lifecycle = client.get_resolution_state(str(raw.get("conditionId")))
            if lifecycle:
                status = str(lifecycle[0].get("status") or "").lower()
                if status not in {"initialized", "active"}:
                    exclusions.append(
                        {
                            "market_id": candidate.market_id,
                            "question": candidate.question,
                            "reasons": ["resolution lifecycle is not unresolved: {}".format(status or "unknown")],
                            "resolution_state": lifecycle,
                            "raw": raw,
                        }
                    )
                    continue
        except Exception as exc:
            exclusions.append(
                {
                    "market_id": candidate.market_id,
                    "question": candidate.question,
                    "reasons": ["resolution state could not be verified: {}: {}".format(type(exc).__name__, exc)],
                    "raw": raw,
                }
            )
            continue
        try:
            book = client.get_order_book(candidate.yes_asset_id)
            quote = quote_from_order_book(
                book,
                utc_now(),
                config.max_quote_age_seconds,
                config.max_spread,
            )
        except Exception as exc:
            exclusions.append(
                {
                    "market_id": candidate.market_id,
                    "question": candidate.question,
                    "reasons": ["order-book request failed: {}".format(type(exc).__name__)],
                    "raw": raw,
                }
            )
            continue
        if quote.status != "valid":
            exclusions.append(
                {
                    "market_id": candidate.market_id,
                    "question": candidate.question,
                    "reasons": [quote.issue or quote.status],
                    "quote": quote.__dict__,
                    "raw": raw,
                }
            )
            continue
        entry = candidate.to_dict()
        entry["discovery_quote"] = quote.__dict__
        entry["selection_metadata"] = {
            "liquidity_usd": candidate.liquidity,
            "volume_usd": candidate.volume,
            "discovery_yes_midpoint": quote.midpoint,
            "discovery_spread": quote.spread,
        }
        candidates.append(entry)
        if len(candidates) >= max_candidates:
            break
    report = {
        "report_version": 1,
        "created_at": iso_utc(utc_now()),
        "study_id": config.study_id,
        "mode": config.mode,
        "study_deadline": config.study_deadline,
        "selection_criteria": {
            "binary_yes_no": True,
            "active_and_unresolved_at_enrollment": True,
            "min_liquidity_usd": config.min_liquidity_usd,
            "min_volume_usd": config.min_volume_usd,
            "max_spread": config.max_spread,
            "max_quote_age_seconds": config.max_quote_age_seconds,
            "end_date_before_study_deadline": True,
            "requires_resolution_rules_or_source": True,
            "quote_method": "YES token best-bid/best-ask midpoint",
        },
        "eligible_candidates": candidates,
        "exclusions": exclusions,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = str(Path(report_path).with_suffix(".csv"))
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["market_id", "event_id", "question", "expected_end_at", "category", "liquidity", "volume", "url"],
        )
        writer.writeheader()
        for item in candidates:
            writer.writerow({key: item.get(key) for key in writer.fieldnames})
    return report


def MarketCandidateFromRaw(raw: Dict[str, Any], config: StudyConfig) -> Tuple[Optional[MarketCandidate], List[str]]:
    from .polymarket import market_to_candidate

    return market_to_candidate(
        raw,
        config.study_deadline,
        config.min_liquidity_usd,
        config.min_volume_usd,
    )


def enroll(
    report_path: str,
    output_path: str,
    config: StudyConfig,
    count: int = 50,
    random_select: bool = False,
    seed: Optional[int] = None,
    market_api: Optional[Any] = None,
) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    if report.get("study_id") != config.study_id:
        raise ValueError("Candidate report study_id does not match the current config")
    if report.get("mode") != config.mode:
        raise ValueError("Candidate report data mode does not match the current config")
    if report.get("study_deadline") != config.study_deadline:
        raise ValueError("Candidate report deadline differs from current config")
    eligible = list(report.get("eligible_candidates", []))
    exclusions = list(report.get("exclusions", []))
    if market_api is not None:
        still_eligible: List[Dict[str, Any]] = []
        from .polymarket import quote_from_order_book

        for item in eligible:
            try:
                current = market_api.get_market(str(item["market_id"]))
                if current.get("active") is not True or current.get("closed") is not False:
                    raise ValueError("market is no longer active and open")
                lifecycle = market_api.get_resolution_state(
                    str(item.get("raw", {}).get("conditionId") or "")
                )
                if lifecycle and str(lifecycle[0].get("status", "")).lower() not in {
                    "initialized",
                    "active",
                }:
                    raise ValueError(
                        "resolution lifecycle is no longer unresolved: {}".format(
                            lifecycle[0].get("status")
                        )
                    )
                quote = quote_from_order_book(
                    market_api.get_order_book(str(item["yes_asset_id"])),
                    utc_now(),
                    config.max_quote_age_seconds,
                    config.max_spread,
                )
                if quote.status != "valid":
                    raise ValueError("YES quote no longer meets criteria: {}".format(quote.issue))
                item["enrollment_quote"] = quote.__dict__
                metadata = item.setdefault("selection_metadata", {})
                metadata["enrollment_yes_midpoint"] = quote.midpoint
                metadata["enrollment_spread"] = quote.spread
                still_eligible.append(item)
            except Exception as exc:
                exclusions.append(
                    {
                        "market_id": item.get("market_id"),
                        "question": item.get("question"),
                        "reasons": ["enrollment recheck failed: {}: {}".format(type(exc).__name__, exc)],
                    }
                )
        eligible = still_eligible
    chosen: List[Dict[str, Any]] = []
    seen_events = set()
    rng_seed = config.random_seed if seed is None else seed
    rng = random.Random(rng_seed)
    if random_select:
        rng.shuffle(eligible)
    for item in eligible:
        event_key = item.get("event_group") or item.get("event_id") or item.get("market_id")
        if event_key in seen_events:
            exclusions.append(
                {
                    "market_id": item.get("market_id"),
                    "question": item.get("question"),
                    "reasons": ["duplicate/closely related market in the same event group"],
                }
            )
            continue
        seen_events.add(event_key)
        chosen.append(item)
        if len(chosen) >= max(1, count):
            break
    payload: Dict[str, Any] = {
        "study_id": config.study_id,
        "mode": config.mode,
        "manifest_version": config.manifest_version,
        "created_at": iso_utc(utc_now()),
        "study_deadline": config.study_deadline,
        "study_config": config.__dict__,
        "selection_criteria": report.get("selection_criteria", {}),
        "selection": {
            "requested_count": count,
            "enrolled_count": len(chosen),
            "random_selection": random_select,
            "random_seed": rng_seed if random_select else None,
            "event_group_deduplication": "at most one market per event group",
            "exclusions": exclusions,
        },
        "markets": chosen,
    }
    manifest_hash = canonical_hash(payload)
    payload["content_sha256"] = manifest_hash
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def verify_manifest(manifest: Dict[str, Any]) -> None:
    expected = manifest.get("content_sha256")
    payload = dict(manifest)
    payload.pop("content_sha256", None)
    actual = canonical_hash(payload)
    if not expected or actual != expected:
        raise ValueError("Manifest hash verification failed; refusing to use an edited manifest")


def register_manifest(
    connection: Any, manifest: Dict[str, Any], config: StudyConfig
) -> None:
    verify_manifest(manifest)
    study_id = str(manifest["study_id"])
    existing = connection.execute(
        "SELECT manifest_sha256 FROM studies WHERE study_id=?", (study_id,)
    ).fetchone()
    if existing is not None:
        if existing["manifest_sha256"] != manifest["content_sha256"]:
            raise ValueError("Study ID already exists with a different frozen manifest")
        return
    connection.execute(
        "INSERT INTO studies(study_id,mode,deadline,manifest_version,manifest_json,"
        "manifest_sha256,config_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (
            study_id,
            manifest["mode"],
            manifest["study_deadline"],
            manifest["manifest_version"],
            json_dumps(manifest),
            manifest["content_sha256"],
            json_dumps(manifest.get("study_config", config.__dict__)),
            manifest["created_at"],
        ),
    )
    for item in manifest.get("markets", []):
        connection.execute(
            "INSERT INTO markets(study_id,market_id,event_id,event_group,question,resolution_rules,"
            "yes_asset_id,no_asset_id,yes_outcome_index,no_outcome_index,url,category,enrolled_at,"
            "expected_end_at,selection_json,raw_market_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                study_id,
                str(item["market_id"]),
                item.get("event_id"),
                item.get("event_group"),
                item["question"],
                item["resolution_rules"],
                item["yes_asset_id"],
                item["no_asset_id"],
                item["yes_outcome_index"],
                item["no_outcome_index"],
                item["url"],
                item.get("category"),
                manifest["created_at"],
                item["expected_end_at"],
                json_dumps(item.get("selection_metadata", {})),
                json_dumps(item.get("raw", {})),
            ),
        )
    connection.commit()


def manifest_from_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    verify_manifest(manifest)
    return manifest
