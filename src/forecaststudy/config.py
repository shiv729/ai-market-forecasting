from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

from .models import StudyConfig


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(path: str = "config/study.json") -> StudyConfig:
    config_path = Path(path)
    if config_path.exists():
        _load_dotenv(Path(".env"))
        raw: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("Configuration file not found: {}".format(config_path))
    allowed = {item.name for item in fields(StudyConfig)}
    config = StudyConfig(**{key: value for key, value in raw.items() if key in allowed})
    config.model = os.getenv("OPENAI_MODEL", config.model)
    config.study_deadline = os.getenv("STUDY_DEADLINE", config.study_deadline)
    if os.getenv("MAX_DAILY_SPEND_USD"):
        config.max_daily_spend_usd = float(os.environ["MAX_DAILY_SPEND_USD"])
    if os.getenv("ESTIMATED_INPUT_USD_PER_MILLION_TOKENS"):
        config.estimated_input_usd_per_million_tokens = float(
            os.environ["ESTIMATED_INPUT_USD_PER_MILLION_TOKENS"]
        )
    if os.getenv("ESTIMATED_OUTPUT_USD_PER_MILLION_TOKENS"):
        config.estimated_output_usd_per_million_tokens = float(
            os.environ["ESTIMATED_OUTPUT_USD_PER_MILLION_TOKENS"]
        )
    config.validate()
    return config
