from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObservationPose:
    observer_cell: tuple[int, int]
    heading_deg: float


@dataclass(frozen=True)
class AblationScenario:
    scenario_id: str
    width: int
    height: int
    resolution: float
    observations: tuple[ObservationPose, ...]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    elapsed_time: float
    recency_time_constant: float
    low_confidence_band: tuple[int, int]
    value_region: tuple[int, int, int, int]
    risk_region: tuple[int, int, int, int] | None = None
    occlusion_obstacles: tuple[tuple[int, int], ...] = tuple()
    use_simple_occlusion: bool = False
    lookahead_steps: int = 1


def default_ablation_scenario_config_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / "ablation" / "scenarios.json"


def _int_pair(raw: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise ValueError(f"{field_name} must be a two-item cell")
    return (int(raw[0]), int(raw[1]))


def _cell(raw: Any, field_name: str, width: int, height: int) -> tuple[int, int]:
    x, y = _int_pair(raw, field_name)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{field_name} must be inside the scenario grid")
    return (x, y)


def _region(raw: Any, field_name: str, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError(f"{field_name} must be [x0, x1, y0, y1]")
    x0, x1, y0, y1 = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{field_name} must be inside the scenario grid")
    return (x0, x1, y0, y1)


def _band(raw: Any, field_name: str, width: int) -> tuple[int, int]:
    x0, x1 = _int_pair(raw, field_name)
    if not (0 <= x0 < x1 <= width):
        raise ValueError(f"{field_name} must be inside the scenario grid")
    return (x0, x1)


def _positive_float(raw: Any, field_name: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _parse_observations(raw: dict[str, Any], width: int, height: int) -> tuple[ObservationPose, ...]:
    observations_raw = raw.get("observations")
    if observations_raw is None:
        observations_raw = [{"observer_cell": raw.get("observer_cell"), "heading_deg": raw.get("heading_deg", 0.0)}]
    if not isinstance(observations_raw, list) or not observations_raw:
        raise ValueError("observations must be a non-empty list")

    observations: list[ObservationPose] = []
    for index, observation in enumerate(observations_raw):
        if not isinstance(observation, dict):
            raise ValueError("each observation must be an object")
        observations.append(
            ObservationPose(
                observer_cell=_cell(observation.get("observer_cell"), f"observations[{index}].observer_cell", width, height),
                heading_deg=float(observation.get("heading_deg", 0.0)),
            )
        )
    return tuple(observations)


def _parse_scenario(raw: dict[str, Any]) -> AblationScenario:
    width = int(raw.get("width", 0))
    height = int(raw.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    resolution = _positive_float(raw.get("resolution", 0.0), "resolution")
    observations = _parse_observations(raw, width, height)
    risk_region_raw = raw.get("risk_region")
    return AblationScenario(
        scenario_id=str(raw["scenario_id"]),
        width=width,
        height=height,
        resolution=resolution,
        observations=observations,
        start_cell=_cell(raw.get("start_cell"), "start_cell", width, height),
        goal_cell=_cell(raw.get("goal_cell"), "goal_cell", width, height),
        elapsed_time=float(raw.get("elapsed_time", 1.0)),
        recency_time_constant=_positive_float(raw.get("recency_time_constant", 10.0), "recency_time_constant"),
        low_confidence_band=_band(raw.get("low_confidence_band"), "low_confidence_band", width),
        value_region=_region(raw.get("value_region"), "value_region", width, height),
        risk_region=_region(risk_region_raw, "risk_region", width, height) if risk_region_raw is not None else None,
        occlusion_obstacles=tuple(_cell(cell, "occlusion_obstacles", width, height) for cell in raw.get("occlusion_obstacles", ())),
        use_simple_occlusion=bool(raw.get("use_simple_occlusion", False)),
        lookahead_steps=max(1, int(raw.get("lookahead_steps", 1))),
    )


def load_ablation_scenarios(path: str | Path) -> tuple[AblationScenario, ...]:
    """读取确定性消融场景配置，并完成尺寸、位姿和区域边界检查。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    scenarios_raw = raw.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("scenarios must be a non-empty list")
    scenarios = tuple(_parse_scenario(item) for item in scenarios_raw)
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario_id values must be unique")
    return scenarios
