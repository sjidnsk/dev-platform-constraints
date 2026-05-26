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
    rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key not in {"reason_counts"}
    )
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
      <tr><th>路径是否可达</th><td>{escape(str(summary["path_reachable"]))}</td></tr>
      <tr><th>路径节点数</th><td>{escape(str(summary["path_nodes"]))}</td></tr>
      <tr><th>路径总代价</th><td>{escape(str(summary["path_total_cost"]))}</td></tr>
      <tr><th>硬约束违规数</th><td>{escape(str(summary["hard_constraint_violations"]))}</td></tr>
    </tbody>
  </table>
  <h2>完整摘要</h2>
  <table>
    <tbody>
      {rows}
    </tbody>
  </table>
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
    _write_html_report(html_path, image_path, title, summary, reason_counts)
    return VisualizationReport(image_path=image_path, html_path=html_path, summary=summary)
