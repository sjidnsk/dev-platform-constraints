from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import ConstraintResult
from .map_layers import GridMap, metadata_for_generated_layer
from .platform_model import PlatformParameters


@dataclass(frozen=True)
class CostWeights:
    min_cell_cost: float = 1.0
    blocked_cell_cost: float = 1_000_000.0
    slope: float = 4.0
    roughness: float = 2.0
    obstacle: float = 5.0
    illumination: float = 1.0
    confidence: float = 2.0
    value_reward: float = 0.0


def generate_costmap(
    grid: GridMap,
    constraints: ConstraintResult,
    platform: PlatformParameters,
    weights: CostWeights | None = None,
) -> GridMap:
    """生成非负代价图层，以及与代价分离的通行性图层。

    可信度只按已有地形风险比例增加风险项，不直接作为通行性使用，
    以保持设计文档中的边界。
    """

    weights = weights or CostWeights()
    slope = np.nan_to_num(np.asarray(grid.require_layer("slope"), dtype=float), nan=platform.max_slope_deg)
    roughness = np.nan_to_num(np.asarray(grid.require_layer("roughness"), dtype=float), nan=1.0)
    obstacle = np.nan_to_num(np.asarray(grid.require_layer("obstacle"), dtype=float), nan=1.0)
    illumination = np.nan_to_num(np.asarray(grid.require_layer("illumination"), dtype=float), nan=0.0)
    confidence = np.nan_to_num(np.asarray(grid.require_layer("confidence"), dtype=float), nan=0.0)
    value = np.nan_to_num(np.asarray(grid.require_layer("value"), dtype=float), nan=0.0)

    slope_risk = np.clip(slope / max(platform.max_slope_deg, 1e-9), 0.0, 1.0)
    roughness_risk = np.clip(roughness, 0.0, 1.0)
    obstacle_risk = np.clip(obstacle, 0.0, 1.0)
    illumination_risk = np.clip(1.0 - illumination, 0.0, 1.0)
    base_risk = np.clip((slope_risk + roughness_risk + obstacle_risk + illumination_risk) / 4.0, 0.0, 1.0)
    confidence_risk = np.clip((1.0 - confidence) * base_risk, 0.0, 1.0)
    bounded_value_reward = np.clip(value, 0.0, 1.0) * max(weights.value_reward, 0.0)

    cost = (
        weights.min_cell_cost
        + weights.slope * slope_risk
        + weights.roughness * roughness_risk
        + weights.obstacle * obstacle_risk
        + weights.illumination * illumination_risk
        + weights.confidence * confidence_risk
        - bounded_value_reward
    )
    cost = np.maximum(0.0, cost)
    cost = np.where(constraints.passable_mask, cost, weights.blocked_cell_cost)

    risk_score = np.clip(
        0.40 * slope_risk
        + 0.25 * roughness_risk
        + 0.25 * obstacle_risk
        + 0.10 * illumination_risk
        + 0.20 * confidence_risk,
        0.0,
        1.0,
    )
    traversability = np.where(constraints.passable_mask, np.clip(1.0 - risk_score, 0.0, 1.0), 0.0)

    grid.add_layer(
        "cost",
        cost,
        metadata_for_generated_layer("cost", grid.resolution, grid.frame_id, unit="cost"),
    )
    grid.add_layer(
        "traversability",
        traversability,
        metadata_for_generated_layer("traversability", grid.resolution, grid.frame_id, unit="unitless"),
    )
    return grid
