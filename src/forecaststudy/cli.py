from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .analysis import evaluate
from .collection import collect, resolve_study
from .config import load_config
from .db import connect
from .demo import run_demo
from .polymarket import PolymarketClient
from .reporting import export_study
from .study import build_candidate_report, enroll, register_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-study",
        description="Collect and evaluate a preregistered AI-versus-market forecasting study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="find and report eligible Polymarket candidates")
    discover.add_argument("--config", default="config/study.json")
    discover.add_argument("--output", default=None, help="candidate report JSON path")
    discover.add_argument("--max-pages", type=int, default=5)
    discover.add_argument("--max-candidates", type=int, default=200)

    enrollment = subparsers.add_parser("enroll", help="freeze a reviewed candidate report into a manifest")
    enrollment.add_argument("--config", default="config/study.json")
    enrollment.add_argument("--candidates", required=True)
    enrollment.add_argument("--output", default=None)
    enrollment.add_argument("--count", type=int, default=50)
    enrollment.add_argument("--random", action="store_true", help="select randomly with a recorded seed")
    enrollment.add_argument("--seed", type=int, default=None)

    collection = subparsers.add_parser("collect", help="collect one daily forecast per enrolled market")
    collection.add_argument("--config", default="config/study.json")
    collection.add_argument("--manifest", default=None)
    collection.add_argument("--database", default=None)
    collection.add_argument("--dry-run", action="store_true", help="read quotes but make no research/model calls")

    resolution = subparsers.add_parser("resolve", help="refresh and append market outcome status revisions")
    resolution.add_argument("--config", default="config/study.json")
    resolution.add_argument("--manifest", default=None)
    resolution.add_argument("--database", default=None)

    evaluation = subparsers.add_parser("evaluate", help="calculate Brier scores and market-level bootstrap")
    evaluation.add_argument("--config", default="config/study.json")
    evaluation.add_argument("--study-id", default=None)
    evaluation.add_argument("--database", default=None)
    evaluation.add_argument("--bootstrap", type=int, default=None)
    evaluation.add_argument("--seed", type=int, default=None)
    evaluation.add_argument("--event-clusters", action="store_true")

    export = subparsers.add_parser("export", help="write CSV data and a Markdown research report")
    export.add_argument("--config", default="config/study.json")
    export.add_argument("--study-id", default=None)
    export.add_argument("--database", default=None)
    export.add_argument("--output-dir", default=None)

    demo = subparsers.add_parser("demo", help="run a fully synthetic collect/resolve/evaluate/export demonstration")
    demo.add_argument("--data-dir", default="data/demo")
    demo.add_argument("--bootstrap", type=int, default=1000)
    return parser


def _manifest_path(config: Any, requested: Optional[str]) -> str:
    return requested or str(Path(config.data_dir) / "study-manifest.json")


def _database_path(config: Any, requested: Optional[str]) -> str:
    return requested or str(Path(config.data_dir) / "research.sqlite")


def _do_discover(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = args.output or str(Path(config.data_dir) / "candidate-report.json")
    client = PolymarketClient(max_retries=config.max_retries)
    try:
        rows = client.list_markets(
            config.study_deadline,
            config.min_liquidity_usd,
            config.min_volume_usd,
            max_pages=args.max_pages,
        )
        report = build_candidate_report(
            rows, config, client, output, max_candidates=args.max_candidates
        )
    finally:
        client.close()
    print(
        json.dumps(
            {
                "report": output,
                "csv": str(Path(output).with_suffix(".csv")),
                "eligible": len(report["eligible_candidates"]),
                "excluded": len(report["exclusions"]),
                "mode": report["mode"],
            },
            indent=2,
        )
    )
    return 0


def _do_enroll(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = args.output or str(Path(config.data_dir) / "study-manifest.json")
    client = PolymarketClient(max_retries=config.max_retries)
    try:
        manifest = enroll(
            args.candidates, output, config, args.count, args.random, args.seed, client
        )
    finally:
        client.close()
    connection = connect(_database_path(config, None))
    register_manifest(connection, manifest, config)
    connection.close()
    print(
        json.dumps(
            {
                "manifest": output,
                "sha256": manifest["content_sha256"],
                "enrolled": manifest["selection"]["enrolled_count"],
                "requested": args.count,
                "warning": (
                    "fewer eligible markets were available; criteria were not relaxed"
                    if manifest["selection"]["enrolled_count"] < args.count
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


def _do_collect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = collect(
        _manifest_path(config, args.manifest),
        config,
        database_path=_database_path(config, args.database),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0 if not result.get("errors") else 1


def _do_resolve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = resolve_study(
        _manifest_path(config, args.manifest),
        config,
        database_path=_database_path(config, args.database),
    )
    print(json.dumps(result, indent=2))
    return 0 if not result.get("failed") else 1


def _do_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result, _ = evaluate(
        _database_path(config, args.database),
        args.study_id or config.study_id,
        bootstrap_replicates=args.bootstrap or config.bootstrap_replicates,
        bootstrap_seed=args.seed if args.seed is not None else config.bootstrap_seed,
        event_clusters=args.event_clusters,
        config_mode=config.mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _do_export(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = export_study(
        _database_path(config, args.database),
        args.study_id or config.study_id,
        args.output_dir or str(Path(config.data_dir) / "exports"),
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.bootstrap_seed,
        config_mode=config.mode,
    )
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            return _do_discover(args)
        if args.command == "enroll":
            return _do_enroll(args)
        if args.command == "collect":
            return _do_collect(args)
        if args.command == "resolve":
            return _do_resolve(args)
        if args.command == "evaluate":
            return _do_evaluate(args)
        if args.command == "export":
            return _do_export(args)
        if args.command == "demo":
            result = run_demo(args.data_dir, args.bootstrap)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print("error: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
