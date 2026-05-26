from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.confidence import default_confidence_config_path, load_confidence_weights, update_confidence_from_observation
from dev_platform_constraints.core import validate_grid_map
from dev_platform_constraints.exploration import generate_exploration_candidates, rank_exploration_goals
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters
from dev_platform_constraints.reporting import build_data_contract_report, render_closure_report
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最小闭环静态可视化报告。")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "visualization"), help="可视化输出目录。")
    parser.add_argument("--platform", default="yutu2", help="configs/platforms 下的平台配置名称。")
    parser.add_argument("--width", type=int, default=32, help="示例栅格宽度。")
    parser.add_argument("--height", type=int, default=20, help="示例栅格高度。")
    parser.add_argument("--resolution", type=float, default=0.5, help="示例栅格分辨率，单位为米。")
    parser.add_argument("--confidence-config", default=str(default_confidence_config_path()), help="可信度融合权重配置路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = generate_sample_grid(width=args.width, height=args.height, resolution=args.resolution)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = load_platform_parameters(default_platform_config_path(args.platform))
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
    constraints = generate_hard_constraints(grid, platform)
    generate_costmap(grid, constraints, platform)
    validation_report = validate_grid_map(grid)
    candidates = generate_exploration_candidates(
        grid,
        constraints,
        start=(0, 0),
        platform=platform,
        max_candidates=8,
    )
    scored_goals = rank_exploration_goals(candidates)[:5]
    plan = astar_path(
        grid.layers["cost"],
        constraints.passable_mask,
        start=(0, 0),
        goal=(grid.width - 1, grid.height - 1),
        resolution=grid.resolution,
    )

    report = render_closure_report(
        grid,
        constraints,
        plan,
        Path(args.output_dir),
        title=f"最小闭环可视化 - {platform.name}",
        confidence_update_report=confidence_report,
        data_contract_report=build_data_contract_report(grid, validation_report),
        scored_goals=scored_goals,
    )
    summary = dict(report.summary)
    summary["platform"] = platform.name
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not plan.reachable:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
