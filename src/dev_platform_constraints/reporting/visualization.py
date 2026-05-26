from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..core.layers import GridMap
from ..confidence import ConfidenceUpdateReport
from ..mapping.constraints import ConstraintResult
from ..path_planning.astar import PlanningResult


@dataclass(frozen=True)
class VisualizationReport:
    image_path: Path
    html_path: Path
    summary: dict[str, object]


def _reason_counts(constraints: ConstraintResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code, name in sorted(constraints.reason_names.items()):
        counts[name] = int(np.count_nonzero(np.bitwise_and(constraints.reason_layer, code)))
    return counts


def _finite_min_max(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def _add_layer_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    data: np.ndarray,
    title: str,
    cmap: str | ListedColormap,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> None:
    image = axis.imshow(np.asarray(data, dtype=float), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    if colorbar:
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _add_path(axis: plt.Axes, plan: PlanningResult) -> None:
    if not plan.path:
        return
    xs = [point[0] for point in plan.path]
    ys = [point[1] for point in plan.path]
    axis.plot(xs, ys, color="red", linewidth=2.0, label="path")
    axis.scatter([xs[0]], [ys[0]], color="lime", edgecolors="black", s=60, label="start", zorder=3)
    axis.scatter([xs[-1]], [ys[-1]], color="dodgerblue", edgecolors="black", s=60, label="goal", zorder=3)
    axis.legend(loc="upper right", fontsize=8)


def _low_confidence_high_risk_path_ratio(
    grid: GridMap,
    plan: PlanningResult,
    confidence_threshold: float = 0.5,
    risk_threshold: float = 0.5,
) -> float:
    if not plan.path:
        return 0.0
    confidence = np.asarray(grid.require_layer("confidence"), dtype=float)
    roughness = np.clip(np.asarray(grid.require_layer("roughness"), dtype=float), 0.0, 1.0)
    obstacle = np.clip(np.asarray(grid.require_layer("obstacle"), dtype=float), 0.0, 1.0)
    illumination_risk = np.clip(1.0 - np.asarray(grid.require_layer("illumination"), dtype=float), 0.0, 1.0)
    base_risk = np.clip((roughness + obstacle + illumination_risk) / 3.0, 0.0, 1.0)

    risky_nodes = 0
    for x, y in plan.path:
        if confidence[y, x] < confidence_threshold and base_risk[y, x] >= risk_threshold:
            risky_nodes += 1
    return float(risky_nodes / len(plan.path))


def _write_html_report(
    html_path: Path,
    image_path: Path,
    title: str,
    summary: dict[str, object],
    reason_counts: dict[str, int],
) -> None:
    def table_rows(rows: tuple[tuple[str, object], ...]) -> str:
        return "\n".join(
            f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
            for label, value in rows
        )

    path_rows = table_rows(
        (
            ("路径是否可达", summary["path_reachable"]),
            ("路径节点数", summary["path_nodes"]),
            ("路径总代价", summary["path_total_cost"]),
            ("扩展节点数", summary["expanded_nodes"]),
            ("硬约束违规数", summary["hard_constraint_violations"]),
            ("低可信高风险路径比例", summary["low_confidence_high_risk_path_ratio"]),
        )
    )
    grid_rows = table_rows(
        (
            ("栅格宽度", summary["grid_width"]),
            ("栅格高度", summary["grid_height"]),
            ("栅格分辨率", summary["grid_resolution"]),
            ("可通行代价最小值", summary["cost_min"]),
            ("可通行代价最大值", summary["cost_max"]),
        )
    )
    confidence_section = ""
    if "confidence_delta_c" in summary:
        confidence_rows = table_rows(
            (
                ("更新前平均可信度", summary["confidence_mean_before"]),
                ("更新后平均可信度", summary["confidence_mean_after"]),
                ("平均可信度变化", summary["confidence_mean_delta"]),
                ("更新前低可信区域面积", summary["confidence_low_area_before"]),
                ("更新后低可信区域面积", summary["confidence_low_area_after"]),
                ("可信度正向提升总量", summary["confidence_delta_c"]),
                ("传感器可见栅格数", summary["confidence_visible_cell_count"]),
                ("已更新栅格数", summary["confidence_updated_cell_count"]),
            )
        )
        confidence_section = f"""
  <h2>可信度更新摘要</h2>
  <table>
    <tbody>
      {confidence_rows}
    </tbody>
  </table>"""

    data_contract = summary.get("data_contract")
    contract_section = ""
    if isinstance(data_contract, dict):
        issue_summary = data_contract.get("issue_summary", {})
        layers = data_contract.get("layers", {})
        missing_layers = data_contract.get("missing_layers", ())
        contract_rows = table_rows(
            (
                ("数据契约是否有效", data_contract.get("is_valid")),
                ("图层数量", len(layers) if isinstance(layers, dict) else 0),
                ("缺失核心图层数", len(missing_layers) if isinstance(missing_layers, (list, tuple)) else 0),
                ("契约错误数", issue_summary.get("errors") if isinstance(issue_summary, dict) else None),
                ("契约警告数", issue_summary.get("warnings") if isinstance(issue_summary, dict) else None),
            )
        )
        contract_section = f"""
  <h2>数据契约摘要</h2>
  <table>
    <tbody>
      {contract_rows}
    </tbody>
  </table>"""

    reason_rows = "\n".join(
        f"<tr><th>{escape(name)}</th><td>{count}</td></tr>"
        for name, count in reason_counts.items()
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 32px; color: #222; line-height: 1.5; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #555; margin-top: 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; min-width: 420px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="subtitle">平台、路径和硬约束摘要。</p>
  <h2>总览图</h2>
  <img src="{escape(image_path.name)}" alt="最小闭环可视化总览图">
  <h2>路径与约束摘要</h2>
  <table>
    <tbody>
      {path_rows}
    </tbody>
  </table>
  <h2>地图与代价摘要</h2>
  <table>
    <tbody>
      {grid_rows}
    </tbody>
  </table>
  {confidence_section}
  {contract_section}
  <h2>约束原因计数</h2>
  <table>
    <tbody>
      {reason_rows}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def render_closure_report(
    grid: GridMap,
    constraints: ConstraintResult,
    plan: PlanningResult,
    output_dir: str | Path,
    title: str = "最小闭环可视化",
    confidence_update_report: ConfidenceUpdateReport | None = None,
    data_contract_report: dict[str, object] | None = None,
) -> VisualizationReport:
    """渲染最小闭环的静态 PNG 总览图和 HTML 报告。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = output_path / "minimal_closure.png"
    html_path = output_path / "minimal_closure.html"

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    flat_axes = axes.ravel()
    _add_layer_panel(figure, flat_axes[0], grid.require_layer("elevation"), "elevation", "terrain")
    _add_layer_panel(figure, flat_axes[1], grid.require_layer("slope"), "slope", "magma")
    _add_layer_panel(figure, flat_axes[2], grid.require_layer("roughness"), "roughness", "viridis", 0.0, 1.0)
    _add_layer_panel(figure, flat_axes[3], grid.require_layer("obstacle"), "obstacle", "Reds", 0.0, 1.0)
    _add_layer_panel(figure, flat_axes[4], grid.require_layer("confidence"), "confidence", "Blues", 0.0, 1.0)
    _add_layer_panel(figure, flat_axes[5], grid.require_layer("traversability"), "traversability", "Greens", 0.0, 1.0)

    visible_cost = np.where(constraints.passable_mask, grid.require_layer("cost"), np.nan)
    _add_layer_panel(figure, flat_axes[6], visible_cost, "cost + path", "viridis")
    _add_path(flat_axes[6], plan)

    hard_constraint_view = np.where(constraints.passable_mask, 0.0, 1.0)
    _add_layer_panel(
        figure,
        flat_axes[7],
        hard_constraint_view,
        "hard constraints",
        ListedColormap(["#f2f2f2", "#d73027"]),
        0.0,
        1.0,
        colorbar=False,
    )

    figure.savefig(image_path, dpi=150)
    plt.close(figure)

    cost_min, cost_max = _finite_min_max(np.asarray(visible_cost, dtype=float))
    reason_counts = _reason_counts(constraints)
    summary: dict[str, Any] = {
        "image_path": str(image_path),
        "html_path": str(html_path),
        "grid_width": grid.width,
        "grid_height": grid.height,
        "grid_resolution": grid.resolution,
        "path_reachable": plan.reachable,
        "path_nodes": len(plan.path),
        "path_total_cost": float(plan.total_cost) if np.isfinite(plan.total_cost) else None,
        "expanded_nodes": plan.expanded_nodes,
        "hard_constraint_violations": constraints.violation_count,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "reason_counts": reason_counts,
        "low_confidence_high_risk_path_ratio": _low_confidence_high_risk_path_ratio(grid, plan),
    }
    if confidence_update_report is not None:
        summary.update(
            {
                "confidence_mean_before": confidence_update_report.mean_confidence_before,
                "confidence_mean_after": confidence_update_report.mean_confidence_after,
                "confidence_mean_delta": confidence_update_report.mean_confidence_delta,
                "confidence_low_area_before": confidence_update_report.low_confidence_area_before,
                "confidence_low_area_after": confidence_update_report.low_confidence_area_after,
                "confidence_delta_c": confidence_update_report.delta_c,
                "confidence_visible_cell_count": confidence_update_report.visible_cell_count,
                "confidence_updated_cell_count": confidence_update_report.updated_cell_count,
            }
        )
    if data_contract_report is not None:
        summary["data_contract"] = data_contract_report
    _write_html_report(html_path, image_path, title, summary, reason_counts)
    return VisualizationReport(image_path=image_path, html_path=html_path, summary=summary)
