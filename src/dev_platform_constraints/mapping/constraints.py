from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from ..core.layers import GridMap
from ..platforms.model import PlatformParameters


class ReasonCode(IntFlag):
    NONE = 0
    INVALID = 1
    SLOPE = 2
    OBSTACLE = 4
    OBSTACLE_HEIGHT = 8
    FORBIDDEN = 16


REASON_NAMES = {
    ReasonCode.INVALID: "invalid",
    ReasonCode.SLOPE: "slope",
    ReasonCode.OBSTACLE: "obstacle",
    ReasonCode.OBSTACLE_HEIGHT: "obstacle_height",
    ReasonCode.FORBIDDEN: "forbidden",
}


@dataclass(frozen=True)
class ConstraintResult:
    passable_mask: np.ndarray
    reason_layer: np.ndarray
    reason_names: dict[int, str]

    @property
    def violation_count(self) -> int:
        return int(np.count_nonzero(self.reason_layer))


def _add_reason(reason_layer: np.ndarray, mask: np.ndarray, reason: ReasonCode) -> None:
    reason_layer[mask] = np.bitwise_or(reason_layer[mask], int(reason))


def generate_hard_constraints(
    grid: GridMap,
    platform: PlatformParameters,
    obstacle_threshold: float = 0.5,
    forbidden_threshold: float = 0.5,
) -> ConstraintResult:
    """生成 P0 硬约束掩膜和可解释的原因位图层。"""

    shape = grid.shape
    reason_layer = np.zeros(shape, dtype=np.uint16)
    valid_mask = grid.layers.get("valid_mask", np.ones(shape, dtype=bool)).astype(bool, copy=False)
    _add_reason(reason_layer, ~valid_mask, ReasonCode.INVALID)

    slope = np.asarray(grid.require_layer("slope"), dtype=float)
    _add_reason(reason_layer, np.isfinite(slope) & (slope > platform.max_slope_deg), ReasonCode.SLOPE)
    _add_reason(reason_layer, ~np.isfinite(slope), ReasonCode.INVALID)

    obstacle = np.asarray(grid.require_layer("obstacle"), dtype=float)
    _add_reason(reason_layer, np.isfinite(obstacle) & (obstacle >= obstacle_threshold), ReasonCode.OBSTACLE)
    _add_reason(reason_layer, ~np.isfinite(obstacle), ReasonCode.INVALID)

    obstacle_height = np.asarray(grid.require_layer("obstacle_height"), dtype=float)
    height_limit = min(platform.max_obstacle_height, platform.ground_clearance)
    _add_reason(
        reason_layer,
        np.isfinite(obstacle_height) & (obstacle_height > height_limit),
        ReasonCode.OBSTACLE_HEIGHT,
    )
    _add_reason(reason_layer, ~np.isfinite(obstacle_height), ReasonCode.INVALID)

    if "forbidden" in grid.layers:
        forbidden = np.asarray(grid.layers["forbidden"], dtype=float)
        _add_reason(reason_layer, np.isfinite(forbidden) & (forbidden >= forbidden_threshold), ReasonCode.FORBIDDEN)
        _add_reason(reason_layer, ~np.isfinite(forbidden), ReasonCode.INVALID)

    passable = reason_layer == int(ReasonCode.NONE)
    reason_names = {int(code): name for code, name in REASON_NAMES.items()}
    return ConstraintResult(passable_mask=passable, reason_layer=reason_layer, reason_names=reason_names)
