from __future__ import annotations

import numpy as np

from .map_layers import GridMap, metadata_for_generated_layer


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
