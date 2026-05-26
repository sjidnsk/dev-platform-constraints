from __future__ import annotations

import numpy as np

from ..core.layers import GridMap, derived_metadata


def _effective_valid_mask(elevation: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    finite = np.isfinite(elevation)
    if valid_mask is None:
        return finite
    return finite & valid_mask.astype(bool, copy=False)


def _fill_invalid_with_nearest_mean(elevation: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros_like(elevation, dtype=float)
    mean_value = float(np.nanmean(elevation[valid]))
    return np.where(valid, elevation, mean_value).astype(float)


def derive_slope(
    elevation: np.ndarray,
    resolution: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从 2.5D 高程栅格派生以度为单位的坡度。

    边界栅格使用 NumPy 的一阶边缘梯度。无效栅格会从返回的有效掩膜中排除，
    并在坡度图层中写入 `NaN`。
    """

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    elevation_array = np.asarray(elevation, dtype=float)
    valid = _effective_valid_mask(elevation_array, valid_mask)
    filled = _fill_invalid_with_nearest_mean(elevation_array, valid)
    dz_dy, dz_dx = np.gradient(filled, resolution, resolution, edge_order=1)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope = slope.astype(float)
    slope[~valid] = np.nan
    return slope, valid.copy()


def derive_roughness(
    elevation: np.ndarray,
    window_size: int = 3,
    normalization_height: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用局部高程标准差派生归一化崎岖度。"""

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if normalization_height <= 0.0:
        raise ValueError("normalization_height must be positive")

    elevation_array = np.asarray(elevation, dtype=float)
    valid = _effective_valid_mask(elevation_array, valid_mask)
    radius = window_size // 2
    roughness = np.full(elevation_array.shape, np.nan, dtype=float)
    roughness_valid = np.zeros(elevation_array.shape, dtype=bool)

    for row in range(elevation_array.shape[0]):
        row_start = max(0, row - radius)
        row_end = min(elevation_array.shape[0], row + radius + 1)
        for col in range(elevation_array.shape[1]):
            if not valid[row, col]:
                continue
            col_start = max(0, col - radius)
            col_end = min(elevation_array.shape[1], col + radius + 1)
            local_values = elevation_array[row_start:row_end, col_start:col_end]
            local_valid = valid[row_start:row_end, col_start:col_end]
            finite_values = local_values[local_valid]
            if finite_values.size == 0:
                continue
            roughness[row, col] = min(float(np.std(finite_values) / normalization_height), 1.0)
            roughness_valid[row, col] = True

    return roughness, roughness_valid


def derive_terrain_features(
    grid: GridMap,
    roughness_window_size: int = 3,
    roughness_normalization_height: float = 1.0,
) -> GridMap:
    elevation = grid.require_layer("elevation")
    valid_mask = grid.layers.get("valid_mask")
    slope, slope_valid = derive_slope(elevation, grid.resolution, valid_mask=valid_mask)
    roughness, roughness_valid = derive_roughness(
        elevation,
        window_size=roughness_window_size,
        normalization_height=roughness_normalization_height,
        valid_mask=valid_mask,
    )

    elevation_metadata = grid.layer_metadata("elevation")
    grid.add_layer("slope", slope, derived_metadata("elevation", elevation_metadata, unit="deg"))
    grid.add_layer("roughness", roughness, derived_metadata("elevation", elevation_metadata, unit="unitless"))

    if valid_mask is not None:
        grid.layers["valid_mask"] = valid_mask.astype(bool, copy=True) & slope_valid & roughness_valid
    return grid
