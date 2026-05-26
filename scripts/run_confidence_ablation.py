from __future__ import annotations

import argparse
import csv
import json
import sys
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
from dev_platform_constraints.reporting.visualization import _low_confidence_high_risk_path_ratio
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


def _configure_plot_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行可信度权重消融实验。")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "ablation"), help="消融报告输出目录。")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            str(default_confidence_config_path()),
            str(ROOT / "configs" / "confidence" / "observation_focused.json"),
            str(ROOT / "configs" / "confidence" / "consistency_recency_focused.json"),
        ],
        help="需要比较的可信度权重配置路径列表。",
    )
    parser.add_argument("--top-k", type=int, default=3, help="每次实验保留的探索目标数量。")
    return parser.parse_args()


def _run_single_config(config_path: Path, top_k: int) -> dict[str, object]:
    grid = generate_sample_grid(width=32, height=20, resolution=0.5)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = load_platform_parameters(default_platform_config_path("yutu2"))
    weights = load_confidence_weights(config_path)
    confidence_report = update_confidence_from_observation(
        grid,
        platform,
        observer_cell=(0, grid.height // 2),
        heading_deg=0.0,
        elapsed_time=2.0,
        recency_time_constant=10.0,
        weights=weights,
    )
    constraints = generate_hard_constraints(grid, platform)
    generate_costmap(grid, constraints, platform)
    validation_report = validate_grid_map(grid)
    plan = astar_path(
        grid.layers["cost"],
        constraints.passable_mask,
        start=(0, 0),
        goal=(grid.width - 1, grid.height - 1),
        resolution=grid.resolution,
    )
    candidates = generate_exploration_candidates(grid, constraints, start=(0, 0), platform=platform, max_candidates=12)
    scored_goals = rank_exploration_goals(candidates)[: max(top_k, 0)]

    return {
        "confidence_config": config_path.name,
        "validation_valid": validation_report.is_valid,
        "path_reachable": plan.reachable,
        "path_nodes": len(plan.path),
        "path_total_cost": float(plan.total_cost) if plan.reachable else None,
        "confidence_mean_before": confidence_report.mean_confidence_before,
        "confidence_mean_after": confidence_report.mean_confidence_after,
        "confidence_mean_delta": confidence_report.mean_confidence_delta,
        "confidence_low_area_before": confidence_report.low_confidence_area_before,
        "confidence_low_area_after": confidence_report.low_confidence_area_after,
        "confidence_delta_c": confidence_report.delta_c,
        "low_confidence_high_risk_path_ratio": _low_confidence_high_risk_path_ratio(grid, plan),
        "top_goal_cells": [goal.candidate.cell for goal in scored_goals],
        "top_goal_utilities": [goal.utility for goal in scored_goals],
    }


def _write_csv(path: Path, runs: list[dict[str, object]]) -> None:
    fieldnames = [
        "confidence_config",
        "validation_valid",
        "path_reachable",
        "path_nodes",
        "path_total_cost",
        "confidence_mean_before",
        "confidence_mean_after",
        "confidence_mean_delta",
        "confidence_low_area_before",
        "confidence_low_area_after",
        "confidence_delta_c",
        "low_confidence_high_risk_path_ratio",
        "top_goal_cells",
        "top_goal_utilities",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row["top_goal_cells"] = json.dumps(row["top_goal_cells"], ensure_ascii=False)
            row["top_goal_utilities"] = json.dumps(row["top_goal_utilities"], ensure_ascii=False)
            writer.writerow(row)


def _float_series(runs: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        raw = run.get(key)
        values.append(float(raw) if raw is not None else 0.0)
    return values


def _write_ablation_image(path: Path, runs: list[dict[str, object]]) -> None:
    """绘制可信度权重消融的核心指标对比图。"""

    _configure_plot_font()
    labels = [str(run["confidence_config"]).replace(".json", "") for run in runs]
    x_positions = range(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    charts = (
        (axes[0, 0], "路径总代价", _float_series(runs, "path_total_cost"), "#4c78a8"),
        (axes[0, 1], "可信度正向提升总量 ΔC", _float_series(runs, "confidence_delta_c"), "#59a14f"),
        (axes[1, 0], "更新后低可信区域面积", _float_series(runs, "confidence_low_area_after"), "#f28e2b"),
        (axes[1, 1], "低可信高风险路径比例", _float_series(runs, "low_confidence_high_risk_path_ratio"), "#e15759"),
    )
    for axis, title, values, color in charts:
        axis.bar(x_positions, values, color=color)
        axis.set_title(title)
        axis.set_xticks(list(x_positions), labels, rotation=20, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("可信度权重消融指标对比", fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _format_goal_cells(cells: object) -> str:
    if not isinstance(cells, list):
        return str(cells)
    return ", ".join(f"({cell[0]}, {cell[1]})" for cell in cells if isinstance(cell, (list, tuple)) and len(cell) == 2)


def _write_ablation_html(path: Path, image_path: Path, runs: list[dict[str, object]]) -> None:
    """写入可信度权重消融的中文 HTML 报告。"""

    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(run['confidence_config']))}</td>"
        f"<td>{escape(str(run['path_total_cost']))}</td>"
        f"<td>{escape(str(run['confidence_delta_c']))}</td>"
        f"<td>{escape(str(run['confidence_low_area_after']))}</td>"
        f"<td>{escape(str(run['low_confidence_high_risk_path_ratio']))}</td>"
        f"<td>{escape(_format_goal_cells(run['top_goal_cells']))}</td>"
        "</tr>"
        for run in runs
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>可信度权重消融报告</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 32px; color: #222; line-height: 1.5; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #555; margin-top: 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; min-width: 760px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>可信度权重消融报告</h1>
  <p class="subtitle">对比不同可信度融合权重对路径、可信度更新和探索目标排序的影响。</p>
  <h2>指标对比图</h2>
  <img src="{escape(image_path.name)}" alt="可信度权重消融指标对比图">
  <h2>实验摘要</h2>
  <table>
    <thead>
      <tr><th>配置</th><th>路径总代价</th><th>可信度正向提升总量</th><th>更新后低可信区域面积</th><th>低可信高风险路径比例</th><th>Top-K 探索目标</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [_run_single_config(Path(path), args.top_k) for path in args.configs]
    json_path = output_dir / "confidence_ablation.json"
    csv_path = output_dir / "confidence_ablation.csv"
    image_path = output_dir / "confidence_ablation.png"
    html_path = output_dir / "confidence_ablation.html"
    summary = {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "image_path": str(image_path),
        "html_path": str(html_path),
        "runs": runs,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, runs)
    _write_ablation_image(image_path, runs)
    _write_ablation_html(html_path, image_path, runs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(not run["validation_valid"] or not run["path_reachable"] for run in runs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
