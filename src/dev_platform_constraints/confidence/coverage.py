from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.layers import GridMap, metadata_for_generated_layer
from ..platforms.model import PlatformParameters
from .metrics import compute_observation_model


@dataclass(frozen=True)
class CoverageEstimate:
    total_valid_cell_count: int
    covered_valid_cell_count: int
    visible_cell_count: int
    newly_covered_cell_count: int
    total_valid_area: float
    covered_valid_area: float
    newly_covered_area: float
    coverage_rate: float
    expected_coverage_rate_delta: float
    coverage_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CoverageUpdateReport:
    total_valid_cell_count: int
    covered_valid_cell_count_before: int
    covered_valid_cell_count: int
    newly_covered_cell_count: int
    total_valid_area: float
    covered_valid_area_before: float
    covered_valid_area: float
    newly_covered_area: float
    coverage_rate_before: float
    coverage_rate: float
    coverage_rate_delta: float
    visible_cell_count: int


def _valid_mask(grid: GridMap) -> np.ndarray:
    return grid.layers.get("valid_mask", np.ones(grid.shape, dtype=bool)).astype(bool, copy=False)


def _coverage_cells(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(x), int(y)) for y, x in np.argwhere(mask))


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def ensure_coverage_mask(grid: GridMap, layer_name: str = "coverage_mask") -> np.ndarray:
    """确保地图存在累计覆盖掩膜图层，缺省时初始化为全未覆盖。"""

    if layer_name in grid.layers:
        coverage = np.asarray(grid.layers[layer_name], dtype=bool)
        if coverage.shape != grid.shape:
            raise ValueError(f"layer {layer_name!r} shape must match grid shape")
        if coverage.dtype != grid.layers[layer_name].dtype:
            grid.add_layer(
                layer_name,
                coverage,
                metadata_for_generated_layer(
                    layer_name,
                    grid.resolution,
                    grid.frame_id,
                    unit="boolean",
                    source_kind="coverage",
                ),
            )
        return grid.layers[layer_name].astype(bool, copy=False)

    coverage = np.zeros(grid.shape, dtype=bool)
    grid.add_layer(
        layer_name,
        coverage,
        metadata_for_generated_layer(
            layer_name,
            grid.resolution,
            grid.frame_id,
            unit="boolean",
            source_kind="coverage",
        ),
    )
    return grid.layers[layer_name].astype(bool, copy=False)


def estimate_observation_coverage(
    grid: GridMap,
    platform: PlatformParameters,
    observer_cell: tuple[int, int],
    heading_deg: float,
    *,
    use_simple_occlusion: bool = False,
    layer_name: str = "coverage_mask",
) -> CoverageEstimate:
    """估计一次观测会带来的新增有效地图覆盖。"""

    valid = _valid_mask(grid)
    coverage = ensure_coverage_mask(grid, layer_name) & valid
    model = compute_observation_model(
        grid,
        platform,
        observer_cell,
        heading_deg,
        use_simple_occlusion=use_simple_occlusion,
    )
    visible = np.asarray(model.visible_mask, dtype=bool) & valid
    newly_covered = visible & ~coverage

    cell_area = grid.resolution * grid.resolution
    total_count = int(np.count_nonzero(valid))
    covered_count = int(np.count_nonzero(coverage))
    visible_count = int(np.count_nonzero(visible))
    newly_covered_count = int(np.count_nonzero(newly_covered))
    total_area = float(total_count * cell_area)
    covered_area = float(covered_count * cell_area)
    newly_covered_area = float(newly_covered_count * cell_area)
    return CoverageEstimate(
        total_valid_cell_count=total_count,
        covered_valid_cell_count=covered_count,
        visible_cell_count=visible_count,
        newly_covered_cell_count=newly_covered_count,
        total_valid_area=total_area,
        covered_valid_area=covered_area,
        newly_covered_area=newly_covered_area,
        coverage_rate=_rate(covered_count, total_count),
        expected_coverage_rate_delta=_rate(newly_covered_count, total_count),
        coverage_cells=_coverage_cells(visible),
    )


def update_coverage_from_observation(
    grid: GridMap,
    platform: PlatformParameters,
    observer_cell: tuple[int, int],
    heading_deg: float = 0.0,
    *,
    use_simple_occlusion: bool = False,
    layer_name: str = "coverage_mask",
) -> CoverageUpdateReport:
    """应用一次观测覆盖更新，累计记录唯一有效地图覆盖。"""

    valid = _valid_mask(grid)
    before = ensure_coverage_mask(grid, layer_name) & valid
    estimate = estimate_observation_coverage(
        grid,
        platform,
        observer_cell,
        heading_deg,
        use_simple_occlusion=use_simple_occlusion,
        layer_name=layer_name,
    )
    after = before.copy()
    for x, y in estimate.coverage_cells:
        after[y, x] = True
    after &= valid
    grid.add_layer(
        layer_name,
        after,
        metadata_for_generated_layer(
            layer_name,
            grid.resolution,
            grid.frame_id,
            unit="boolean",
            source_kind="coverage_update",
        ),
    )

    total_count = estimate.total_valid_cell_count
    covered_before_count = estimate.covered_valid_cell_count
    covered_count = int(np.count_nonzero(after))
    cell_area = grid.resolution * grid.resolution
    total_area = float(total_count * cell_area)
    covered_before_area = float(covered_before_count * cell_area)
    covered_area = float(covered_count * cell_area)
    newly_covered_area = float(estimate.newly_covered_cell_count * cell_area)
    return CoverageUpdateReport(
        total_valid_cell_count=total_count,
        covered_valid_cell_count_before=covered_before_count,
        covered_valid_cell_count=covered_count,
        newly_covered_cell_count=estimate.newly_covered_cell_count,
        total_valid_area=total_area,
        covered_valid_area_before=covered_before_area,
        covered_valid_area=covered_area,
        newly_covered_area=newly_covered_area,
        coverage_rate_before=_rate(covered_before_count, total_count),
        coverage_rate=_rate(covered_count, total_count),
        coverage_rate_delta=_rate(estimate.newly_covered_cell_count, total_count),
        visible_cell_count=estimate.visible_cell_count,
    )
