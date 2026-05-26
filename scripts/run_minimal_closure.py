from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.confidence import update_confidence_from_observation
from dev_platform_constraints.core import validate_grid_map
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


def main() -> None:
    grid = generate_sample_grid(width=32, height=20, resolution=0.5)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = load_platform_parameters(default_platform_config_path("yutu2"))
    confidence_report = update_confidence_from_observation(
        grid,
        platform,
        observer_cell=(0, grid.height // 2),
        heading_deg=0.0,
        elapsed_time=2.0,
        recency_time_constant=10.0,
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
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not report.is_valid:
        raise SystemExit(2)
    if not plan.reachable:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
