from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from html import escape
from math import sqrt
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


@dataclass(frozen=True)
class AblationScenario:
    scenario_id: str
    width: int
    height: int
    resolution: float
    observer_cell: tuple[int, int]
    heading_deg: float
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    elapsed_time: float
    recency_time_constant: float
    low_confidence_band: tuple[int, int]
    value_region: tuple[int, int, int, int]


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


def _default_scenarios() -> list[AblationScenario]:
    return [
        AblationScenario(
            scenario_id="baseline_gap",
            width=32,
            height=20,
            resolution=0.5,
            observer_cell=(0, 10),
            heading_deg=0.0,
            start_cell=(0, 0),
            goal_cell=(31, 19),
            elapsed_time=2.0,
            recency_time_constant=10.0,
            low_confidence_band=(10, 12),
            value_region=(27, 32, 16, 20),
        ),
        AblationScenario(
            scenario_id="upper_observation",
            width=32,
            height=20,
            resolution=0.5,
            observer_cell=(0, 6),
            heading_deg=0.0,
            start_cell=(0, 2),
            goal_cell=(31, 15),
            elapsed_time=3.0,
            recency_time_constant=8.0,
            low_confidence_band=(7, 10),
            value_region=(24, 30, 3, 9),
        ),
        AblationScenario(
            scenario_id="compact_value",
            width=24,
            height=16,
            resolution=0.5,
            observer_cell=(0, 8),
            heading_deg=0.0,
            start_cell=(0, 0),
            goal_cell=(23, 15),
            elapsed_time=1.5,
            recency_time_constant=12.0,
            low_confidence_band=(5, 8),
            value_region=(17, 24, 11, 16),
        ),
    ]


def _apply_scenario_layers(grid, scenario: AblationScenario) -> None:
    grid.layers["confidence"][:] = 0.75
    band_x0, band_x1 = scenario.low_confidence_band
    grid.layers["confidence"][:, max(0, band_x0) : min(grid.width, band_x1)] = 0.45
    grid.layers["value"][:] = 0.0
    value_x0, value_x1, value_y0, value_y1 = scenario.value_region
    grid.layers["value"][
        max(0, value_y0) : min(grid.height, value_y1),
        max(0, value_x0) : min(grid.width, value_x1),
    ] = 0.8


def _run_single_config(config_path: Path, scenario: AblationScenario, top_k: int) -> dict[str, object]:
    grid = generate_sample_grid(width=scenario.width, height=scenario.height, resolution=scenario.resolution)
    _apply_scenario_layers(grid, scenario)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = load_platform_parameters(default_platform_config_path("yutu2"))
    weights = load_confidence_weights(config_path)
    confidence_report = update_confidence_from_observation(
        grid,
        platform,
        observer_cell=scenario.observer_cell,
        heading_deg=scenario.heading_deg,
        elapsed_time=scenario.elapsed_time,
        recency_time_constant=scenario.recency_time_constant,
        weights=weights,
    )
    constraints = generate_hard_constraints(grid, platform)
    generate_costmap(grid, constraints, platform)
    validation_report = validate_grid_map(grid)
    plan = astar_path(
        grid.layers["cost"],
        constraints.passable_mask,
        start=scenario.start_cell,
        goal=scenario.goal_cell,
        resolution=grid.resolution,
    )
    candidates = generate_exploration_candidates(grid, constraints, start=scenario.start_cell, platform=platform, max_candidates=12)
    scored_goals = rank_exploration_goals(candidates)[: max(top_k, 0)]

    return {
        "scenario_id": scenario.scenario_id,
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
        "scenario_id",
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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = _mean(values)
    return float(sqrt(sum((value - average) ** 2 for value in values) / len(values)))


def _goal_overlap(reference: object, current: object) -> float:
    if not isinstance(reference, list) or not isinstance(current, list):
        return 0.0
    reference_set = {tuple(cell) for cell in reference if isinstance(cell, (list, tuple)) and len(cell) == 2}
    current_set = {tuple(cell) for cell in current if isinstance(cell, (list, tuple)) and len(cell) == 2}
    if not reference_set and not current_set:
        return 1.0
    if not reference_set or not current_set:
        return 0.0
    return float(len(reference_set & current_set) / len(reference_set | current_set))


def _first_goal(cells: object) -> tuple[int, int] | None:
    if not isinstance(cells, list) or not cells:
        return None
    first = cells[0]
    if isinstance(first, (list, tuple)) and len(first) == 2:
        return (int(first[0]), int(first[1]))
    return None


def _build_aggregate(runs: list[dict[str, object]], scenarios: list[AblationScenario]) -> list[dict[str, object]]:
    configs = list(dict.fromkeys(str(run["confidence_config"]) for run in runs))
    best_counts = {config: 0 for config in configs}
    for scenario in scenarios:
        scenario_runs = [run for run in runs if run["scenario_id"] == scenario.scenario_id and run.get("path_total_cost") is not None]
        if not scenario_runs:
            continue
        best_cost = min(float(run["path_total_cost"]) for run in scenario_runs)
        for run in scenario_runs:
            if abs(float(run["path_total_cost"]) - best_cost) <= 1e-9:
                best_counts[str(run["confidence_config"])] += 1

    aggregate: list[dict[str, object]] = []
    for config in configs:
        config_runs = [run for run in runs if run["confidence_config"] == config]
        path_costs = [float(run["path_total_cost"]) for run in config_runs if run.get("path_total_cost") is not None]
        delta_cs = _float_series(config_runs, "confidence_delta_c")
        low_areas = _float_series(config_runs, "confidence_low_area_after")
        risk_ratios = _float_series(config_runs, "low_confidence_high_risk_path_ratio")
        reference_goals = config_runs[0].get("top_goal_cells") if config_runs else []
        top_goal_stability = _mean([_goal_overlap(reference_goals, run.get("top_goal_cells")) for run in config_runs])
        reference_first = _first_goal(reference_goals)
        first_goal_change_count = sum(1 for run in config_runs if _first_goal(run.get("top_goal_cells")) != reference_first)
        failure_scenarios = [
            str(run["scenario_id"])
            for run in config_runs
            if not run.get("validation_valid") or not run.get("path_reachable")
        ]
        aggregate.append(
            {
                "confidence_config": config,
                "path_total_cost_mean": _mean(path_costs),
                "path_total_cost_std": _std(path_costs),
                "confidence_delta_c_mean": _mean(delta_cs),
                "confidence_delta_c_std": _std(delta_cs),
                "confidence_low_area_after_mean": _mean(low_areas),
                "low_confidence_high_risk_path_ratio_mean": _mean(risk_ratios),
                "best_path_cost_count": best_counts[config],
                "top_goal_stability": top_goal_stability,
                "first_goal_change_count": first_goal_change_count,
                "failure_scenarios": failure_scenarios,
            }
        )
    return aggregate


def _write_ablation_image(path: Path, runs: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    """绘制多场景可信度权重消融的汇总指标对比图。"""

    _configure_plot_font()
    labels = [str(item["confidence_config"]).replace(".json", "") for item in aggregate]
    x_positions = range(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    charts = (
        (axes[0, 0], "路径总代价均值", _float_series(aggregate, "path_total_cost_mean"), "#4c78a8"),
        (axes[0, 1], "可信度正向提升总量 ΔC 均值", _float_series(aggregate, "confidence_delta_c_mean"), "#59a14f"),
        (axes[1, 0], "配置胜率", _float_series(aggregate, "best_path_cost_count"), "#f28e2b"),
        (axes[1, 1], "Top-K 稳定性", _float_series(aggregate, "top_goal_stability"), "#e15759"),
    )
    for axis, title, values, color in charts:
        axis.bar(x_positions, values, color=color)
        axis.set_title(title)
        axis.set_xticks(list(x_positions), labels, rotation=20, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("多场景可信度权重消融汇总", fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _format_goal_cells(cells: object) -> str:
    if not isinstance(cells, list):
        return str(cells)
    return ", ".join(f"({cell[0]}, {cell[1]})" for cell in cells if isinstance(cell, (list, tuple)) and len(cell) == 2)


def _write_ablation_html(
    path: Path,
    image_path: Path,
    runs: list[dict[str, object]],
    aggregate: list[dict[str, object]],
) -> None:
    """写入多场景可信度权重消融的中文 HTML 报告。"""

    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(run['scenario_id']))}</td>"
        f"<td>{escape(str(run['confidence_config']))}</td>"
        f"<td>{escape(str(run['path_total_cost']))}</td>"
        f"<td>{escape(str(run['confidence_delta_c']))}</td>"
        f"<td>{escape(str(run['confidence_low_area_after']))}</td>"
        f"<td>{escape(str(run['low_confidence_high_risk_path_ratio']))}</td>"
        f"<td>{escape(_format_goal_cells(run['top_goal_cells']))}</td>"
        "</tr>"
        for run in runs
    )
    aggregate_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(item['confidence_config']))}</td>"
        f"<td>{float(item['path_total_cost_mean']):.6g}</td>"
        f"<td>{float(item['path_total_cost_std']):.6g}</td>"
        f"<td>{float(item['confidence_delta_c_mean']):.6g}</td>"
        f"<td>{escape(str(item['best_path_cost_count']))}</td>"
        f"<td>{float(item['top_goal_stability']):.3f}</td>"
        f"<td>{escape(str(item['first_goal_change_count']))}</td>"
        f"<td>{escape(', '.join(item['failure_scenarios']) if item['failure_scenarios'] else '无')}</td>"
        "</tr>"
        for item in aggregate
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>多场景可信度权重消融报告</title>
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
  <p class="subtitle">对比不同可信度融合权重在多场景下对路径、可信度更新和探索目标排序的影响。</p>
  <h2>多场景汇总图</h2>
  <img src="{escape(image_path.name)}" alt="可信度权重消融指标对比图">
  <h2>汇总指标</h2>
  <table>
    <thead>
      <tr><th>配置</th><th>路径总代价均值</th><th>路径总代价标准差</th><th>可信度正向提升总量均值</th><th>配置胜率</th><th>Top-K 稳定性</th><th>首选目标变化次数</th><th>失败场景</th></tr>
    </thead>
    <tbody>
      {aggregate_rows}
    </tbody>
  </table>
  <h2>实验摘要</h2>
  <table>
    <thead>
      <tr><th>场景</th><th>配置</th><th>路径总代价</th><th>可信度正向提升总量</th><th>更新后低可信区域面积</th><th>低可信高风险路径比例</th><th>Top-K 探索目标</th></tr>
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
    scenarios = _default_scenarios()
    runs = [
        _run_single_config(Path(path), scenario, args.top_k)
        for scenario in scenarios
        for path in args.configs
    ]
    aggregate = _build_aggregate(runs, scenarios)
    json_path = output_dir / "confidence_ablation.json"
    csv_path = output_dir / "confidence_ablation.csv"
    image_path = output_dir / "confidence_ablation.png"
    html_path = output_dir / "confidence_ablation.html"
    summary = {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "image_path": str(image_path),
        "html_path": str(html_path),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "aggregate": aggregate,
        "runs": runs,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, runs)
    _write_ablation_image(image_path, runs, aggregate)
    _write_ablation_html(html_path, image_path, runs, aggregate)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(not run["validation_valid"] or not run["path_reachable"] for run in runs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
