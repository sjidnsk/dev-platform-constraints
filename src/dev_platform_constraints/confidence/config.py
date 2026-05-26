from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import ConfidenceWeights


def load_confidence_weights(path: str | Path) -> ConfidenceWeights:
    """从 JSON 配置加载可信度融合权重。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)
    weights = raw.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("confidence config must contain a weights object")

    values: dict[str, float] = {}
    for field in ("resolution", "observation", "recency", "consistency", "model"):
        if field not in weights:
            raise ValueError(f"confidence weight {field!r} is missing")
        value = float(weights[field])
        if value < 0.0:
            raise ValueError("confidence weights must be nonnegative")
        values[field] = value

    if sum(values.values()) <= 0.0:
        raise ValueError("at least one confidence weight must be positive")
    return ConfidenceWeights(**values)


def default_confidence_config_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / "confidence" / "default.json"
