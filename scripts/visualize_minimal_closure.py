from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.constraints import generate_hard_constraints
from dev_platform_constraints.costmap import generate_costmap
from dev_platform_constraints.examples import generate_sample_grid
from dev_platform_constraints.planning import astar_path
from dev_platform_constraints.platform_model import default_platform_config_path, load_platform_parameters
from dev_platform_constraints.terrain_features import derive_terrain_features
from dev_platform_constraints.visualization import render_closure_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最小闭环静态可视化报告。")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "visualization"), help="可视化输出目录。")
    parser.add_argument("--platform", default="yutu2", help="configs/platforms 下的平台配置名称。")
    parser.add_argument("--width", type=int, default=32, help="示例栅格宽度。")
    parser.add_argument("--height", type=int, default=20, help="示例栅格高度。")
    parser.add_argument("--resolution", type=float, default=0.5, help="示例栅格分辨率，单位为米。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = generate_sample_grid(width=args.width, height=args.height, resolution=args.resolution)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = load_platform_parameters(default_platform_config_path(args.platform))
    constraints = generate_hard_constraints(grid, platform)
    generate_costmap(grid, constraints, platform)
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
    )
    summary = dict(report.summary)
    summary["platform"] = platform.name
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not plan.reachable:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
