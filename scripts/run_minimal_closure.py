from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.confidence import (
    default_confidence_config_path,
    load_confidence_weights,
    update_confidence_from_observation,
    update_coverage_from_observation,
)
from dev_platform_constraints.core import validate_grid_map
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters
from dev_platform_constraints.reporting import build_data_contract_report
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行最小环境-平台-约束闭环。")
    parser.add_argument("--confidence-config", default=str(default_confidence_config_path()), help="可信度融合权重配置路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = generate_sample_grid(width=32, height=20, resolution=0.5)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = load_platform_parameters(default_platform_config_path("yutu2"))
    confidence_weights = load_confidence_weights(args.confidence_config)
    confidence_report = update_confidence_from_observation(
        grid,
        platform,
        observer_cell=(0, grid.height // 2),
        heading_deg=0.0,
        elapsed_time=2.0,
        recency_time_constant=10.0,
        weights=confidence_weights,
    )
    coverage_report = update_coverage_from_observation(
        grid,
        platform,
        observer_cell=(0, grid.height // 2),
        heading_deg=0.0,
    )
    constraints = generate_hard_constraints(grid, platform)
    generate_costmap(grid, constraints, platform)
    report = validate_grid_map(grid)
    plan = astar_path(
        grid.layers["cost"],
        constraints.passable_mask,
        start=(0, 0),
        goal=(grid.width - 1, grid.height - 1),
        resolution=grid.resolution,
    )

    summary = {
        "grid": {"width": grid.width, "height": grid.height, "resolution": grid.resolution},
        "platform": platform.name,
        "validation_valid": report.is_valid,
        "validation": report.format(),
        "data_contract": build_data_contract_report(grid, report),
        "hard_constraint_violations": constraints.violation_count,
        "path_reachable": plan.reachable,
        "path_nodes": len(plan.path),
        "path_total_cost": plan.total_cost,
        "expanded_nodes": plan.expanded_nodes,
        "cost_min": float(grid.layers["cost"].min()),
        "cost_max": float(grid.layers["cost"].max()),
        "traversability_min": float(grid.layers["traversability"].min()),
        "traversability_max": float(grid.layers["traversability"].max()),
        "confidence_mean_before": confidence_report.mean_confidence_before,
        "confidence_mean_after": confidence_report.mean_confidence_after,
        "confidence_mean_delta": confidence_report.mean_confidence_delta,
        "confidence_low_area_before": confidence_report.low_confidence_area_before,
        "confidence_low_area_after": confidence_report.low_confidence_area_after,
        "confidence_delta_c": confidence_report.delta_c,
        "confidence_visible_cell_count": confidence_report.visible_cell_count,
        "confidence_updated_cell_count": confidence_report.updated_cell_count,
        "total_valid_area": coverage_report.total_valid_area,
        "covered_valid_area": coverage_report.covered_valid_area,
        "newly_covered_area": coverage_report.newly_covered_area,
        "coverage_rate": coverage_report.coverage_rate,
        "coverage_rate_delta": coverage_report.coverage_rate_delta,
        "total_valid_cell_count": coverage_report.total_valid_cell_count,
        "covered_valid_cell_count": coverage_report.covered_valid_cell_count,
        "newly_covered_cell_count": coverage_report.newly_covered_cell_count,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not report.is_valid:
        raise SystemExit(2)
    if not plan.reachable:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
