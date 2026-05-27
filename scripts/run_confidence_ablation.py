from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from html import escape
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.confidence import (
    compute_terrain_category_likelihood,
    default_confidence_config_path,
    derive_confidence_from_categorical_posterior,
    load_confidence_weights,
    update_categorical_posterior,
    update_confidence_from_observation,
)
from dev_platform_constraints.core import validate_grid_map
from dev_platform_constraints.exploration import evaluate_goal_sequences, generate_exploration_candidates, rank_exploration_goals
from dev_platform_constraints.experiments import AblationScenario, default_ablation_scenario_config_path, load_ablation_scenarios
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters
from dev_platform_constraints.reporting.visualization import _low_confidence_high_risk_path_ratio
from dev_platform_constraints.sample_data import generate_sample_grid, generate_seeded_synthetic_grid, load_npz_grid
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
    parser.add_argument(
        "--scenario-config",
        default=str(default_ablation_scenario_config_path()),
        help="确定性消融场景配置 JSON。",
    )
    parser.add_argument("--top-k", type=int, default=3, help="每次实验保留的探索目标数量。")
    return parser.parse_args()


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
    for x, y in scenario.occlusion_obstacles:
        grid.layers["obstacle"][y, x] = 1.0
        grid.layers["obstacle_height"][y, x] = 0.28


def _build_scenario_grid(scenario: AblationScenario):
    if scenario.map_source.kind == "seeded_synthetic":
        return generate_seeded_synthetic_grid(
            width=scenario.width,
            height=scenario.height,
            resolution=scenario.resolution,
            seed=int(scenario.map_source.seed or 0),
        )
    if scenario.map_source.kind == "npz_grid":
        if scenario.map_source.path is None:
            raise ValueError("map_source.path is required for npz_grid")
        grid = load_npz_grid(scenario.map_source.path)
        if grid.width != scenario.width or grid.height != scenario.height:
            raise ValueError("npz_grid dimensions must match scenario width and height")
        if abs(grid.resolution - scenario.resolution) > 1e-9:
            raise ValueError("npz_grid resolution must match scenario resolution")
        return grid
    return generate_sample_grid(width=scenario.width, height=scenario.height, resolution=scenario.resolution)


def _apply_scenario_risk_layers(grid, scenario: AblationScenario) -> None:
    if scenario.risk_region is None:
        return
    x0, x1, y0, y1 = scenario.risk_region
    grid.layers["roughness"][y0:y1, x0:x1] = 1.0
    grid.layers["illumination"][y0:y1, x0:x1] = 0.0
    grid.layers["confidence"][y0:y1, x0:x1] = np.minimum(grid.layers["confidence"][y0:y1, x0:x1], 0.25)


def _apply_observation_updates(grid, platform, scenario: AblationScenario, weights) -> dict[str, float | int]:
    before = grid.require_layer("confidence").copy()
    visible_cell_count = 0
    updated_cell_count = 0
    for observation in scenario.observations:
        report = update_confidence_from_observation(
            grid,
            platform,
            observer_cell=observation.observer_cell,
            heading_deg=observation.heading_deg,
            elapsed_time=scenario.elapsed_time,
            recency_time_constant=scenario.recency_time_constant,
            weights=weights,
        )
        visible_cell_count += report.visible_cell_count
        updated_cell_count += report.updated_cell_count

    after = grid.require_layer("confidence")
    valid_mask = grid.layers.get("valid_mask", np.ones(grid.shape, dtype=bool)).astype(bool, copy=False)
    cell_area = grid.resolution * grid.resolution
    before_valid = before[valid_mask]
    after_valid = after[valid_mask]
    delta = after - before
    return {
        "mean_confidence_before": float(np.mean(before_valid)) if before_valid.size else 0.0,
        "mean_confidence_after": float(np.mean(after_valid)) if after_valid.size else 0.0,
        "mean_confidence_delta": float(np.mean(after_valid) - np.mean(before_valid)) if before_valid.size else 0.0,
        "low_confidence_area_before": float(np.count_nonzero(before_valid < 0.5) * cell_area),
        "low_confidence_area_after": float(np.count_nonzero(after_valid < 0.5) * cell_area),
        "delta_c": float(np.sum(np.maximum(delta[valid_mask], 0.0)) * cell_area),
        "visible_cell_count": visible_cell_count,
        "updated_cell_count": updated_cell_count,
    }


def _terrain_model_confidence_mean(grid) -> float:
    valid_mask = grid.layers.get("valid_mask", np.ones(grid.shape, dtype=bool)).astype(bool, copy=False)
    likelihood = compute_terrain_category_likelihood(
        slope=grid.require_layer("slope"),
        roughness=grid.require_layer("roughness"),
        obstacle=grid.require_layer("obstacle"),
        illumination=grid.require_layer("illumination"),
        valid_mask=valid_mask,
    )
    prior = np.full(likelihood.probabilities.shape, 1.0 / len(likelihood.categories), dtype=float)
    quality = np.clip(np.asarray(grid.require_layer("confidence"), dtype=float), 0.0, 1.0)
    posterior = update_categorical_posterior(
        prior,
        likelihood.probabilities,
        quality,
        categories=likelihood.categories,
        valid_mask=valid_mask,
    )
    confidence = derive_confidence_from_categorical_posterior(posterior)
    valid_values = confidence.values[confidence.valid_mask]
    return float(np.mean(valid_values)) if valid_values.size else 0.0


def _sequence_details(goal_sequences) -> list[dict[str, object]]:
    return [
        {
            "cells": [goal.cell for goal in sequence.goals],
            "utility": sequence.utility,
            "delta_c": sequence.delta_c,
            "value_coverage": sequence.value_coverage,
            "risk": sequence.risk,
            "path_cost": sequence.path_cost,
            "coverage_area": sequence.coverage_area,
            "segment_path_costs": list(sequence.segment_path_costs),
            "cumulative_risk": sequence.cumulative_risk,
            "reachable": sequence.reachable,
            "unreachable_reasons": list(sequence.unreachable_reasons),
        }
        for sequence in goal_sequences
    ]


def _run_single_config(config_path: Path, scenario: AblationScenario, top_k: int) -> dict[str, object]:
    grid = _build_scenario_grid(scenario)
    _apply_scenario_layers(grid, scenario)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    _apply_scenario_risk_layers(grid, scenario)
    terrain_model_confidence_mean = _terrain_model_confidence_mean(grid)
    platform = load_platform_parameters(default_platform_config_path("yutu2"))
    weights = load_confidence_weights(config_path)
    confidence_report = _apply_observation_updates(grid, platform, scenario, weights)
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
    candidates = generate_exploration_candidates(
        grid,
        constraints,
        start=scenario.start_cell,
        platform=platform,
        max_candidates=12,
        lookahead_steps=scenario.lookahead_steps,
        use_simple_occlusion=scenario.use_simple_occlusion,
    )
    scored_goals = rank_exploration_goals(candidates)[: max(top_k, 0)]
    goal_sequences = evaluate_goal_sequences(candidates, depth=3, beam_width=max(top_k, 1))[: max(top_k, 0)]

    return {
        "scenario_id": scenario.scenario_id,
        "confidence_config": config_path.name,
        "map_source_kind": scenario.map_source.kind,
        "map_source_seed": scenario.map_source.seed,
        "validation_valid": validation_report.is_valid,
        "path_reachable": plan.reachable,
        "path_nodes": len(plan.path),
        "path_total_cost": float(plan.total_cost) if plan.reachable else None,
        "observation_count": len(scenario.observations),
        "use_simple_occlusion": scenario.use_simple_occlusion,
        "lookahead_steps": scenario.lookahead_steps,
        "confidence_mean_before": confidence_report["mean_confidence_before"],
        "confidence_mean_after": confidence_report["mean_confidence_after"],
        "confidence_mean_delta": confidence_report["mean_confidence_delta"],
        "confidence_low_area_before": confidence_report["low_confidence_area_before"],
        "confidence_low_area_after": confidence_report["low_confidence_area_after"],
        "confidence_delta_c": confidence_report["delta_c"],
        "terrain_model_confidence_mean": terrain_model_confidence_mean,
        "low_confidence_high_risk_path_ratio": _low_confidence_high_risk_path_ratio(grid, plan),
        "top_goal_cells": [goal.candidate.cell for goal in scored_goals],
        "top_goal_utilities": [goal.utility for goal in scored_goals],
        "top_goal_sequence_cells": [[goal.cell for goal in sequence.goals] for sequence in goal_sequences],
        "top_goal_sequence_utilities": [sequence.utility for sequence in goal_sequences],
        "top_goal_sequence_details": _sequence_details(goal_sequences),
    }


def _write_csv(path: Path, runs: list[dict[str, object]]) -> None:
    fieldnames = [
        "scenario_id",
        "confidence_config",
        "map_source_kind",
        "map_source_seed",
        "observation_count",
        "use_simple_occlusion",
        "lookahead_steps",
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
        "terrain_model_confidence_mean",
        "low_confidence_high_risk_path_ratio",
        "top_goal_cells",
        "top_goal_utilities",
        "top_goal_sequence_cells",
        "top_goal_sequence_utilities",
        "top_goal_sequence_details",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row["top_goal_cells"] = json.dumps(row["top_goal_cells"], ensure_ascii=False)
            row["top_goal_utilities"] = json.dumps(row["top_goal_utilities"], ensure_ascii=False)
            row["top_goal_sequence_cells"] = json.dumps(row["top_goal_sequence_cells"], ensure_ascii=False)
            row["top_goal_sequence_utilities"] = json.dumps(row["top_goal_sequence_utilities"], ensure_ascii=False)
            row["top_goal_sequence_details"] = json.dumps(row["top_goal_sequence_details"], ensure_ascii=False)
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


def _sequence_overlap(reference: object, current: object) -> float:
    if not isinstance(reference, list) or not isinstance(current, list):
        return 0.0
    reference_set = {tuple(tuple(cell) for cell in sequence) for sequence in reference if isinstance(sequence, list)}
    current_set = {tuple(tuple(cell) for cell in sequence) for sequence in current if isinstance(sequence, list)}
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
        risk_conflict_hit_rate = _mean(
            [1.0 if float(run.get("low_confidence_high_risk_path_ratio") or 0.0) > 0.0 else 0.0 for run in config_runs]
        )
        reference_sequences = config_runs[0].get("top_goal_sequence_cells") if config_runs else []
        sequence_goal_stability = _mean([_sequence_overlap(reference_sequences, run.get("top_goal_sequence_cells")) for run in config_runs])
        map_family_counts: dict[str, int] = {}
        for scenario in scenarios:
            family_runs = [
                run
                for run in runs
                if run["scenario_id"] == scenario.scenario_id and run.get("path_total_cost") is not None
            ]
            if not family_runs:
                continue
            best_cost = min(float(run["path_total_cost"]) for run in family_runs)
            matching = [
                run
                for run in family_runs
                if run["confidence_config"] == config and abs(float(run["path_total_cost"]) - best_cost) <= 1e-9
            ]
            if matching:
                map_family_counts[scenario.map_source.kind] = map_family_counts.get(scenario.map_source.kind, 0) + len(matching)
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
                "risk_conflict_hit_rate": risk_conflict_hit_rate,
                "best_path_cost_count": best_counts[config],
                "top_goal_stability": top_goal_stability,
                "sequence_goal_stability": sequence_goal_stability,
                "map_family_best_path_cost_count": map_family_counts,
                "first_goal_change_count": first_goal_change_count,
                "failure_scenarios": failure_scenarios,
            }
        )
    default_stability = float(aggregate[0]["top_goal_stability"]) if aggregate else 0.0
    for item in aggregate:
        item["top_goal_stability_change"] = float(item["top_goal_stability"]) - default_stability
    return aggregate


def _build_recommendation(aggregate: list[dict[str, object]], scenario_count: int) -> dict[str, object]:
    if not aggregate:
        return {
            "confidence_config": None,
            "reason": "没有可用配置，无法形成推荐。",
        }

    def ranking_key(item: dict[str, object]) -> tuple[float, float, float, float, float]:
        failure_count = len(item.get("failure_scenarios", []))
        return (
            -float(failure_count),
            float(item["best_path_cost_count"]),
            float(item["confidence_delta_c_mean"]),
            -float(item["path_total_cost_mean"]),
            float(item["top_goal_stability"]),
        )

    selected = max(aggregate, key=ranking_key)
    reason = (
        f"推荐 {selected['confidence_config']}：配置胜率 "
        f"{selected['best_path_cost_count']}/{scenario_count}，"
        f"ΔC 均值 {float(selected['confidence_delta_c_mean']):.6g}，"
        f"失败场景 {len(selected['failure_scenarios'])} 个。"
    )
    return {
        "confidence_config": selected["confidence_config"],
        "reason": reason,
        "best_path_cost_count": selected["best_path_cost_count"],
        "scenario_count": scenario_count,
        "failure_scenario_count": len(selected["failure_scenarios"]),
        "risk_conflict_hit_rate": selected["risk_conflict_hit_rate"],
        "top_goal_stability": selected["top_goal_stability"],
        "sequence_goal_stability": selected["sequence_goal_stability"],
    }


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


def _format_sequence_details(details: object) -> str:
    if not isinstance(details, list):
        return str(details)
    formatted: list[str] = []
    for index, sequence in enumerate(details[:2], start=1):
        if not isinstance(sequence, dict):
            continue
        cells = _format_goal_cells(sequence.get("cells"))
        formatted.append(
            "S{index}: cells={cells}; utility={utility:.3g}; coverage={coverage:.3g}; risk={risk:.3g}; reachable={reachable}".format(
                index=index,
                cells=cells,
                utility=float(sequence.get("utility") or 0.0),
                coverage=float(sequence.get("coverage_area") or 0.0),
                risk=float(sequence.get("cumulative_risk") or 0.0),
                reachable=sequence.get("reachable"),
            )
        )
    return " | ".join(formatted)


def _write_ablation_html(
    path: Path,
    image_path: Path,
    runs: list[dict[str, object]],
    aggregate: list[dict[str, object]],
    recommendation: dict[str, object],
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
        f"<td>{escape(_format_sequence_details(run['top_goal_sequence_details']))}</td>"
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
        f"<td>{float(item['risk_conflict_hit_rate']):.3f}</td>"
        f"<td>{float(item['top_goal_stability']):.3f}</td>"
        f"<td>{float(item['sequence_goal_stability']):.3f}</td>"
        f"<td>{float(item['top_goal_stability_change']):+.3f}</td>"
        f"<td>{escape(str(item['first_goal_change_count']))}</td>"
        f"<td>{escape(', '.join(item['failure_scenarios']) if item['failure_scenarios'] else '无')}</td>"
        "</tr>"
        for item in aggregate
    )
    recommendation_rows = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
        for label, value in (
            ("推荐配置", recommendation.get("confidence_config")),
            ("推荐理由", recommendation.get("reason")),
            ("配置胜率", f"{recommendation.get('best_path_cost_count')}/{recommendation.get('scenario_count')}"),
            ("风险冲突命中率", recommendation.get("risk_conflict_hit_rate")),
            ("Top-K 稳定性", recommendation.get("top_goal_stability")),
            ("序列目标稳定性", recommendation.get("sequence_goal_stability")),
            ("失败场景数", recommendation.get("failure_scenario_count")),
        )
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
  <h2>推荐结论</h2>
  <table>
    <tbody>
      {recommendation_rows}
    </tbody>
  </table>
  <h2>汇总指标</h2>
  <table>
    <thead>
      <tr><th>配置</th><th>路径总代价均值</th><th>路径总代价标准差</th><th>可信度正向提升总量均值</th><th>配置胜率</th><th>风险冲突命中率</th><th>Top-K 稳定性</th><th>序列目标稳定性</th><th>Top-K 稳定性变化</th><th>首选目标变化次数</th><th>失败场景</th></tr>
    </thead>
    <tbody>
      {aggregate_rows}
    </tbody>
  </table>
  <h2>实验摘要</h2>
  <table>
    <thead>
      <tr><th>场景</th><th>配置</th><th>路径总代价</th><th>可信度正向提升总量</th><th>更新后低可信区域面积</th><th>低可信高风险路径比例</th><th>Top-K 探索目标</th><th>Top 序列解释</th></tr>
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
    scenarios = load_ablation_scenarios(args.scenario_config)
    runs = [
        _run_single_config(Path(path), scenario, args.top_k)
        for scenario in scenarios
        for path in args.configs
    ]
    aggregate = _build_aggregate(runs, scenarios)
    recommendation = _build_recommendation(aggregate, len(scenarios))
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
        "recommendation": recommendation,
        "runs": runs,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, runs)
    _write_ablation_image(image_path, runs, aggregate)
    _write_ablation_html(html_path, image_path, runs, aggregate, recommendation)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(not run["validation_valid"] or not run["path_reachable"] for run in runs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
