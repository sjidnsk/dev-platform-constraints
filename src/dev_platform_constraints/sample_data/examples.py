from __future__ import annotations

import numpy as np

from ..core.layers import GridMap, metadata_for_generated_layer


def generate_sample_grid(width: int = 32, height: int = 20, resolution: float = 0.5) -> GridMap:
    """创建用于 P0 验证的确定性 2.5D 月面近似示例地图。"""

    grid = GridMap(resolution=resolution, origin=(0.0, 0.0), width=width, height=height, frame_id="moon_local")
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx / max(width - 1, 1)
    y_norm = yy / max(height - 1, 1)

    elevation = 0.08 * xx * resolution + 0.03 * np.sin(yy / 2.0)
    crater = np.exp(-(((x_norm - 0.55) ** 2) / 0.025 + ((y_norm - 0.55) ** 2) / 0.05))
    elevation -= 0.18 * crater

    obstacle = np.zeros((height, width), dtype=float)
    wall_x = max(3, width // 2)
    gap_y = height // 2
    obstacle[2 : height - 2, wall_x] = 1.0
    obstacle[max(1, gap_y - 1) : min(height - 1, gap_y + 2), wall_x] = 0.0

    obstacle_height = obstacle * 0.28
    illumination = np.clip(0.75 + 0.20 * np.cos(y_norm * np.pi), 0.0, 1.0)
    confidence = np.full((height, width), 0.75, dtype=float)
    confidence[:, max(1, width // 3) : max(2, width // 3 + 2)] = 0.45
    value = np.zeros((height, width), dtype=float)
    value[max(0, height - 4) : height, max(0, width - 5) : width] = 0.8
    valid_mask = np.ones((height, width), dtype=bool)

    layers = {
        "elevation": (elevation, "m"),
        "obstacle": (obstacle, "probability"),
        "obstacle_height": (obstacle_height, "m"),
        "illumination": (illumination, "unitless"),
        "confidence": (confidence, "unitless"),
        "value": (value, "unitless"),
        "valid_mask": (valid_mask, "boolean"),
    }
    for name, (data, unit) in layers.items():
        grid.add_layer(name, data, metadata_for_generated_layer(name, resolution, grid.frame_id, unit=unit))
    return grid


def generate_seeded_synthetic_grid(width: int, height: int, resolution: float, seed: int) -> GridMap:
    """创建固定随机种子的半合成月面栅格，用于外推验证。"""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    rng = np.random.default_rng(int(seed))
    grid = GridMap(resolution=resolution, origin=(0.0, 0.0), width=width, height=height, frame_id="moon_local")
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx / max(width - 1, 1)
    y_norm = yy / max(height - 1, 1)

    base_slope = rng.uniform(0.02, 0.07) * xx * resolution + rng.uniform(-0.03, 0.03) * yy * resolution
    undulation = 0.025 * np.sin(xx / rng.uniform(2.5, 5.5)) + 0.020 * np.cos(yy / rng.uniform(2.0, 4.5))
    elevation = base_slope + undulation + rng.normal(0.0, 0.006, size=(height, width))

    for _ in range(3):
        center_x = rng.uniform(0.2, 0.85)
        center_y = rng.uniform(0.15, 0.85)
        radius_x = rng.uniform(0.015, 0.045)
        radius_y = rng.uniform(0.015, 0.060)
        depth = rng.uniform(0.04, 0.12)
        crater = np.exp(-(((x_norm - center_x) ** 2) / radius_x + ((y_norm - center_y) ** 2) / radius_y))
        elevation -= depth * crater

    obstacle = np.zeros((height, width), dtype=float)
    rock_count = max(1, (width * height) // 90)
    rock_ys = rng.integers(1, max(2, height - 1), size=rock_count)
    rock_xs = rng.integers(max(2, width // 4), max(3, width - 2), size=rock_count)
    obstacle[rock_ys, rock_xs] = 0.35
    obstacle_height = obstacle * 0.14

    shadow_center = rng.uniform(0.35, 0.75)
    shadow_width = rng.uniform(0.06, 0.14)
    shadow = np.exp(-((y_norm - shadow_center) ** 2) / shadow_width)
    illumination = np.clip(0.82 - 0.30 * shadow + 0.08 * np.cos(x_norm * np.pi), 0.05, 1.0)
    confidence = np.clip(0.70 + rng.normal(0.0, 0.04, size=(height, width)), 0.45, 0.85)
    value = np.zeros((height, width), dtype=float)
    valid_mask = np.ones((height, width), dtype=bool)

    layers = {
        "elevation": (elevation, "m"),
        "obstacle": (obstacle, "probability"),
        "obstacle_height": (obstacle_height, "m"),
        "illumination": (illumination, "unitless"),
        "confidence": (confidence, "unitless"),
        "value": (value, "unitless"),
        "valid_mask": (valid_mask, "boolean"),
    }
    for name, (data, unit) in layers.items():
        grid.add_layer(name, data, metadata_for_generated_layer(name, resolution, grid.frame_id, unit=unit, source_kind="seeded_synthetic"))
    return grid
