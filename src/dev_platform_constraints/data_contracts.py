from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CORE_LAYERS: tuple[str, ...] = (
    "elevation",
    "slope",
    "roughness",
    "obstacle",
    "obstacle_height",
    "illumination",
    "confidence",
    "value",
    "traversability",
    "cost",
    "valid_mask",
)

NORMALIZED_LAYERS = {
    "roughness",
    "obstacle",
    "illumination",
    "confidence",
    "value",
    "traversability",
}

NONNEGATIVE_LAYERS = {
    "slope",
    "roughness",
    "obstacle",
    "obstacle_height",
    "illumination",
    "confidence",
    "value",
    "traversability",
    "cost",
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    layer: str | None
    message: str

    def format(self) -> str:
        target = self.layer if self.layer is not None else "grid"
        return f"{self.severity.upper()} [{target}] {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    missing_layers: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def format(self) -> str:
        if not self.issues and not self.missing_layers:
            return "ValidationReport(valid=True)"
        lines = [issue.format() for issue in self.issues]
        if self.missing_layers:
            lines.append(f"WARNING [grid] missing layers: {', '.join(self.missing_layers)}")
        return "\n".join(lines)


def validate_grid_map(grid: object, required_layers: Iterable[str] = CORE_LAYERS) -> ValidationReport:
    """校验当前栅格地图中已存在图层是否满足共享数据契约。

    `NaN` 表示未知数据，只允许出现在全局 `valid_mask` 为 false 的位置，
    从而保持低可信度和缺失数据之间的边界。
    """

    issues: list[ValidationIssue] = []
    required = tuple(required_layers)
    missing_layers = tuple(layer for layer in required if layer not in grid.layers)

    valid_mask = grid.layers.get("valid_mask")
    if valid_mask is None:
        issues.append(ValidationIssue("error", "valid_mask", "required valid_mask layer is missing"))
        valid_cells = np.ones((grid.height, grid.width), dtype=bool)
    else:
        if valid_mask.shape != (grid.height, grid.width):
            issues.append(ValidationIssue("error", "valid_mask", "shape does not match grid height/width"))
            valid_cells = np.ones((grid.height, grid.width), dtype=bool)
        else:
            valid_cells = valid_mask.astype(bool, copy=False)
        if valid_mask.dtype != np.bool_:
            issues.append(ValidationIssue("error", "valid_mask", "valid_mask must use boolean dtype"))

    for name, layer in grid.layers.items():
        if layer.shape != (grid.height, grid.width):
            issues.append(ValidationIssue("error", name, "shape does not match grid height/width"))
            continue

        metadata = grid.metadata.get(name)
        if metadata is None:
            issues.append(ValidationIssue("error", name, "layer metadata is missing"))
        else:
            if not metadata.source_id:
                issues.append(ValidationIssue("error", name, "metadata.source_id is required"))
            if not metadata.unit:
                issues.append(ValidationIssue("error", name, "metadata.unit is required"))
            if metadata.resolution != grid.resolution:
                issues.append(ValidationIssue("error", name, "metadata resolution must match grid resolution"))
            if metadata.frame_id != grid.frame_id:
                issues.append(ValidationIssue("error", name, "metadata frame_id must match grid frame_id"))

        if name == "valid_mask":
            continue

        numeric = np.asarray(layer, dtype=float)
        finite_or_nan = np.isfinite(numeric) | np.isnan(numeric)
        if not np.all(finite_or_nan):
            issues.append(ValidationIssue("error", name, "layer contains non-finite values other than NaN"))

        valid_values = numeric[valid_cells]
        if np.isnan(valid_values).any():
            issues.append(ValidationIssue("error", name, "NaN is only allowed where valid_mask is false"))

        finite_values = valid_values[np.isfinite(valid_values)]
        if finite_values.size == 0:
            continue

        if name in NONNEGATIVE_LAYERS and np.any(finite_values < 0.0):
            issues.append(ValidationIssue("error", name, "values must be nonnegative"))
        if name in NORMALIZED_LAYERS and (np.any(finite_values < 0.0) or np.any(finite_values > 1.0)):
            issues.append(ValidationIssue("error", name, "values must be in [0, 1]"))

    for missing in missing_layers:
        issues.append(ValidationIssue("warning", missing, "required core layer is missing"))

    return ValidationReport(issues=tuple(issues), missing_layers=missing_layers)
