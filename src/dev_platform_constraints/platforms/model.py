from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_SOURCE_KINDS = {"public", "estimated", "assumed"}


@dataclass(frozen=True)
class ParameterValue:
    value: float | dict[str, float]
    unit: str
    source_kind: str
    note: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any], field_name: str) -> "ParameterValue":
        missing = {"value", "unit", "source_kind"} - raw.keys()
        if missing:
            raise ValueError(f"parameter {field_name!r} is missing keys: {', '.join(sorted(missing))}")
        source_kind = str(raw["source_kind"])
        if source_kind not in VALID_SOURCE_KINDS:
            raise ValueError(f"parameter {field_name!r} has invalid source_kind {source_kind!r}")
        return cls(
            value=raw["value"],
            unit=str(raw["unit"]),
            source_kind=source_kind,
            note=str(raw.get("note", "")),
        )


@dataclass(frozen=True)
class PlatformParameters:
    name: str
    parameters: dict[str, ParameterValue]

    @classmethod
    def minimal(cls, name: str, max_slope_deg: float, max_obstacle_height: float) -> "PlatformParameters":
        return cls(
            name=name,
            parameters={
                "max_slope_deg": ParameterValue(max_slope_deg, "deg", "assumed", "测试或仿真默认值"),
                "max_obstacle_height": ParameterValue(max_obstacle_height, "m", "assumed", "测试或仿真默认值"),
                "ground_clearance": ParameterValue(0.18, "m", "assumed", "测试或仿真默认值"),
                "min_turning_radius": ParameterValue(0.0, "m", "assumed", "P0 栅格 A* 不强制使用"),
                "sensor_range": ParameterValue(5.0, "m", "assumed", "后续观测模型占位参数"),
                "sensor_fov": ParameterValue(60.0, "deg", "assumed", "后续观测模型占位参数"),
            },
        )

    @property
    def max_slope_deg(self) -> float:
        return self.float_value("max_slope_deg")

    @property
    def max_obstacle_height(self) -> float:
        return self.float_value("max_obstacle_height")

    @property
    def ground_clearance(self) -> float:
        return self.float_value("ground_clearance")

    def float_value(self, key: str) -> float:
        value = self.parameters[key].value
        if isinstance(value, dict):
            raise TypeError(f"parameter {key!r} is structured and cannot be read as float")
        return float(value)

    def validate(self) -> list[str]:
        warnings: list[str] = []
        for required in (
            "max_slope_deg",
            "max_obstacle_height",
            "ground_clearance",
            "min_turning_radius",
            "sensor_range",
            "sensor_fov",
            "energy_model",
        ):
            if required not in self.parameters:
                warnings.append(f"missing parameter: {required}")

        positive_fields = ("max_slope_deg", "max_obstacle_height", "ground_clearance", "sensor_range", "sensor_fov")
        for field in positive_fields:
            if field in self.parameters and self.float_value(field) <= 0.0:
                warnings.append(f"{field} should be positive")

        if "max_slope_deg" in self.parameters and self.float_value("max_slope_deg") > 45.0:
            warnings.append("max_slope_deg exceeds conservative lunar rover bounds")
        if "max_obstacle_height" in self.parameters and self.float_value("max_obstacle_height") > self.ground_clearance:
            warnings.append("max_obstacle_height exceeds ground_clearance; verify platform geometry")

        for key, parameter in self.parameters.items():
            if parameter.source_kind == "assumed":
                warnings.append(f"{key} is assumed and should not be treated as a public fact")
        return warnings


def load_platform_parameters(path: str | Path) -> PlatformParameters:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    params = {key: ParameterValue.from_dict(value, key) for key, value in raw["parameters"].items()}
    platform = PlatformParameters(name=str(raw["name"]), parameters=params)
    blocking = [warning for warning in platform.validate() if warning.startswith("missing") or "should be positive" in warning]
    if blocking:
        raise ValueError("; ".join(blocking))
    return platform


def default_platform_config_path(name: str = "yutu2") -> Path:
    root = Path(__file__).resolve().parents[3]
    filename = f"{name}.json"
    return root / "configs" / "platforms" / filename
