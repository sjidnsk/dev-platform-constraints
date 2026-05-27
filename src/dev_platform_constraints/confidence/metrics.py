from __future__ import annotations

from dataclasses import dataclass
from math import exp, radians
from typing import Iterable

import numpy as np

from ..core.layers import GridMap, metadata_for_generated_layer
from ..platforms.model import PlatformParameters


@dataclass(frozen=True)
class ConfidenceComponent:
    name: str
    values: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class ConfidenceWeights:
    resolution: float = 0.20
    observation: float = 0.30
    recency: float = 0.15
    consistency: float = 0.20
    model: float = 0.15

    def weight_for(self, name: str) -> float:
        return max(float(getattr(self, name, 0.0)), 0.0)


@dataclass(frozen=True)
class ConfidenceUpdateReport:
    mean_confidence_before: float
    mean_confidence_after: float
    mean_confidence_delta: float
    low_confidence_area_before: float
    low_confidence_area_after: float
    delta_c: float
    visible_cell_count: int
    updated_cell_count: int


@dataclass(frozen=True)
class ObservationModelResult:
    visible_mask: np.ndarray
    quality_layer: np.ndarray
    confidence_component: ConfidenceComponent


def _as_bool_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError("valid_mask shape must match component shape")
    return result


def _component(name: str, values: np.ndarray, valid_mask: np.ndarray | None = None) -> ConfidenceComponent:
    value_array = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    mask = _as_bool_mask(valid_mask, value_array.shape) & np.isfinite(value_array)
    return ConfidenceComponent(name=name, values=np.where(mask, value_array, 0.0), valid_mask=mask)


def compute_resolution_confidence(
    grid_resolution: float,
    planning_resolution: float,
    shape: tuple[int, int],
) -> ConfidenceComponent:
    """根据地图分辨率和规划尺度计算分辨率可信度。"""

    if grid_resolution <= 0.0:
        raise ValueError("grid_resolution must be positive")
    if planning_resolution <= 0.0:
        raise ValueError("planning_resolution must be positive")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must be positive")
    score = min(1.0, planning_resolution / grid_resolution)
    return _component("resolution", np.full(shape, score, dtype=float))


def compute_observation_confidence(
    grid: GridMap,
    platform: PlatformParameters,
    observer_cell: tuple[int, int],
    heading_deg: float,
) -> ConfidenceComponent:
    """用简化距离和视场角模型计算传感器观测覆盖可信度。"""

    return compute_observation_model(grid, platform, observer_cell, heading_deg).confidence_component


def _grid_line_cells(start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """返回两格之间的 Bresenham 栅格线，用于一阶遮挡近似。"""

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y
    return tuple(cells)


def _apply_simple_occlusion(grid: GridMap, visible: np.ndarray, observer_cell: tuple[int, int]) -> np.ndarray:
    obstacle = grid.layers.get("obstacle")
    if obstacle is None:
        return visible
    obstacle_array = np.asarray(obstacle, dtype=float)
    if obstacle_array.shape != grid.shape:
        return visible

    result = visible.copy()
    for y, x in np.argwhere(visible):
        cell = (int(x), int(y))
        if cell == observer_cell:
            continue
        line = _grid_line_cells(observer_cell, cell)
        blockers = line[1:-1]
        if any(obstacle_array[blocker_y, blocker_x] >= 0.5 for blocker_x, blocker_y in blockers):
            result[y, x] = False
    return result


def compute_observation_model(
    grid: GridMap,
    platform: PlatformParameters,
    observer_cell: tuple[int, int],
    heading_deg: float,
    *,
    use_simple_occlusion: bool = False,
) -> ObservationModelResult:
    """计算观测几何模型，输出可见区域、观测质量层和可信度分量。"""

    sensor_range = platform.float_value("sensor_range")
    sensor_fov = platform.float_value("sensor_fov")
    if sensor_range <= 0.0:
        raise ValueError("sensor_range must be positive")
    if sensor_fov <= 0.0:
        raise ValueError("sensor_fov must be positive")

    ox, oy = observer_cell
    if not (0 <= ox < grid.width and 0 <= oy < grid.height):
        raise ValueError("observer_cell must be inside the grid")

    yy, xx = np.mgrid[0 : grid.height, 0 : grid.width]
    dx = (xx - ox) * grid.resolution
    dy = (yy - oy) * grid.resolution
    distance = np.hypot(dx, dy)
    heading = radians(heading_deg)
    angles = np.arctan2(dy, dx)
    angle_delta = np.arctan2(np.sin(angles - heading), np.cos(angles - heading))
    in_range = distance <= sensor_range
    in_fov = np.abs(np.degrees(angle_delta)) <= sensor_fov / 2.0
    valid_mask = grid.layers.get("valid_mask", np.ones(grid.shape, dtype=bool)).astype(bool, copy=False)
    visible = in_range & in_fov & valid_mask
    if use_simple_occlusion:
        visible = _apply_simple_occlusion(grid, visible, observer_cell)

    range_quality = np.clip(1.0 - distance / sensor_range, 0.0, 1.0)
    angular_quality = np.clip(np.cos(angle_delta), 0.0, 1.0)
    values = np.where(visible, range_quality * angular_quality, 0.0)
    component = _component("observation", values, visible)
    return ObservationModelResult(
        visible_mask=component.valid_mask,
        quality_layer=component.values,
        confidence_component=component,
    )


def compute_recency_confidence(
    shape: tuple[int, int],
    elapsed_time: float,
    time_constant: float,
    valid_mask: np.ndarray | None = None,
) -> ConfidenceComponent:
    """按指数衰减计算时间新鲜度可信度。"""

    if elapsed_time < 0.0:
        raise ValueError("elapsed_time must be nonnegative")
    if time_constant <= 0.0:
        raise ValueError("time_constant must be positive")
    score = exp(-elapsed_time / time_constant)
    return _component("recency", np.full(shape, score, dtype=float), valid_mask)


def compute_consistency_confidence(
    reference: np.ndarray,
    observed: np.ndarray,
    sigma: float,
    valid_mask: np.ndarray | None = None,
) -> ConfidenceComponent:
    """根据两组环境估计差异计算多源一致性可信度。"""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    reference_array = np.asarray(reference, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if reference_array.shape != observed_array.shape:
        raise ValueError("reference and observed must have the same shape")
    mask = _as_bool_mask(valid_mask, reference_array.shape) & np.isfinite(reference_array) & np.isfinite(observed_array)
    error = observed_array - reference_array
    values = np.exp(-(error**2) / (sigma**2))
    return _component("consistency", values, mask)


def fuse_confidence(
    components: Iterable[ConfidenceComponent],
    weights: ConfidenceWeights | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """融合可信度分量，缺失分量只从有效权重中重新归一化。"""

    weights = weights or ConfidenceWeights()
    component_tuple = tuple(components)
    if not component_tuple:
        raise ValueError("at least one confidence component is required")

    shape = component_tuple[0].values.shape
    weighted_sum = np.zeros(shape, dtype=float)
    weight_sum = np.zeros(shape, dtype=float)
    for component in component_tuple:
        values = np.asarray(component.values, dtype=float)
        valid = np.asarray(component.valid_mask, dtype=bool)
        if values.shape != shape or valid.shape != shape:
            raise ValueError("all confidence components must share the same shape")
        weight = weights.weight_for(component.name)
        if weight <= 0.0:
            continue
        weighted_sum += np.where(valid, np.clip(values, 0.0, 1.0) * weight, 0.0)
        weight_sum += np.where(valid, weight, 0.0)

    fused_valid = weight_sum > 0.0
    fused = np.zeros(shape, dtype=float)
    fused[fused_valid] = np.clip(weighted_sum[fused_valid] / weight_sum[fused_valid], 0.0, 1.0)
    return fused, fused_valid


def update_confidence_from_observation(
    grid: GridMap,
    platform: PlatformParameters,
    observer_cell: tuple[int, int],
    heading_deg: float = 0.0,
    planning_resolution: float | None = None,
    elapsed_time: float = 1.0,
    recency_time_constant: float = 10.0,
    weights: ConfidenceWeights | None = None,
    low_confidence_threshold: float = 0.5,
) -> ConfidenceUpdateReport:
    """模拟一次局部观测并更新 `confidence` 图层。"""

    before = np.asarray(grid.require_layer("confidence"), dtype=float)
    valid_mask = grid.layers.get("valid_mask", np.ones(grid.shape, dtype=bool)).astype(bool, copy=False)
    observation = compute_observation_confidence(grid, platform, observer_cell, heading_deg)
    visible = observation.valid_mask

    resolution = compute_resolution_confidence(
        grid.resolution,
        planning_resolution or grid.resolution,
        grid.shape,
    )
    fresh_recency = _component("recency", np.ones(grid.shape, dtype=float), visible)
    fused_visible, fused_visible_valid = fuse_confidence(
        (resolution, observation, fresh_recency),
        weights=weights,
    )

    decay = compute_recency_confidence(grid.shape, elapsed_time, recency_time_constant, valid_mask=valid_mask).values
    after = np.where(valid_mask, np.clip(before * decay, 0.0, 1.0), 0.0)
    update_mask = visible & fused_visible_valid & valid_mask
    after[update_mask] = np.maximum(after[update_mask], fused_visible[update_mask])
    after = np.clip(after, 0.0, 1.0)

    cell_area = grid.resolution * grid.resolution
    before_valid = before[valid_mask]
    after_valid = after[valid_mask]
    low_before = float(np.count_nonzero(before_valid < low_confidence_threshold) * cell_area)
    low_after = float(np.count_nonzero(after_valid < low_confidence_threshold) * cell_area)
    delta = after - before
    report = ConfidenceUpdateReport(
        mean_confidence_before=float(np.mean(before_valid)) if before_valid.size else 0.0,
        mean_confidence_after=float(np.mean(after_valid)) if after_valid.size else 0.0,
        mean_confidence_delta=float(np.mean(after_valid) - np.mean(before_valid)) if before_valid.size else 0.0,
        low_confidence_area_before=low_before,
        low_confidence_area_after=low_after,
        delta_c=float(np.sum(np.maximum(delta[valid_mask], 0.0)) * cell_area),
        visible_cell_count=int(np.count_nonzero(visible & valid_mask)),
        updated_cell_count=int(np.count_nonzero(update_mask)),
    )

    grid.add_layer(
        "confidence",
        after,
        metadata_for_generated_layer(
            "confidence",
            grid.resolution,
            grid.frame_id,
            unit="unitless",
            source_kind="confidence_update",
        ),
    )
    return report
