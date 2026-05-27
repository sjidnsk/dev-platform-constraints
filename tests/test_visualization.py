import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev_platform_constraints.confidence import ConfidenceUpdateReport, CoverageUpdateReport
from dev_platform_constraints.core import validate_grid_map
from dev_platform_constraints.exploration import CandidateGoal, rank_exploration_goals
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import PlatformParameters
from dev_platform_constraints.reporting import build_data_contract_report, render_closure_report
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


def build_minimal_closure():
    grid = generate_sample_grid(width=16, height=10, resolution=0.5)
    derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
    platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
    constraints = generate_hard_constraints(grid, platform)
    generate_costmap(grid, constraints, platform)
    plan = astar_path(
        grid.layers["cost"],
        constraints.passable_mask,
        start=(0, 0),
        goal=(grid.width - 1, grid.height - 1),
        resolution=grid.resolution,
    )
    return grid, constraints, plan


class VisualizationTests(unittest.TestCase):
    def test_render_closure_report_writes_png_html_and_summary(self) -> None:
        grid, constraints, plan = build_minimal_closure()

        output_dir = Path(tempfile.mkdtemp(prefix="dev-platform-constraints-visualization-"))
        data_contract = build_data_contract_report(grid, validate_grid_map(grid))
        confidence_report = ConfidenceUpdateReport(
            mean_confidence_before=0.7,
            mean_confidence_after=0.8,
            mean_confidence_delta=0.1,
            low_confidence_area_before=3.0,
            low_confidence_area_after=1.0,
            delta_c=0.5,
            visible_cell_count=12,
            updated_cell_count=10,
        )
        coverage_report = CoverageUpdateReport(
            total_valid_cell_count=160,
            covered_valid_cell_count_before=10,
            covered_valid_cell_count=22,
            newly_covered_cell_count=12,
            total_valid_area=40.0,
            covered_valid_area_before=2.5,
            covered_valid_area=5.5,
            newly_covered_area=3.0,
            coverage_rate_before=0.0625,
            coverage_rate=0.1375,
            coverage_rate_delta=0.075,
            visible_cell_count=12,
        )
        scored_goals = rank_exploration_goals(
            (
                CandidateGoal(cell=(3, 4), information_gain=0.8, value=0.9, confidence_gain=0.7, risk=0.2, path_cost=4.0),
                CandidateGoal(cell=(5, 6), information_gain=0.4, value=0.2, confidence_gain=0.3, risk=0.1, path_cost=2.0),
            )
        )
        report = render_closure_report(
            grid,
            constraints,
            plan,
            output_dir,
            title="测试可视化",
            confidence_update_report=confidence_report,
            coverage_update_report=coverage_report,
            data_contract_report=data_contract,
            scored_goals=scored_goals,
        )

        self.assertTrue(report.image_path.exists())
        self.assertTrue(report.html_path.exists())
        self.assertGreater(report.image_path.stat().st_size, 0)
        self.assertGreater(report.html_path.stat().st_size, 0)
        self.assertEqual(report.image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("minimal_closure.png", html)
        self.assertIn("测试可视化", html)
        self.assertIn("路径是否可达", html)
        self.assertIn("硬约束违规数", html)
        self.assertIn("可信度更新摘要", html)
        self.assertIn("可信度正向提升总量", html)
        self.assertIn("覆盖率摘要", html)
        self.assertIn("覆盖率增量", html)
        self.assertIn("数据契约摘要", html)
        self.assertIn("契约错误数", html)
        self.assertIn("探索目标摘要", html)
        self.assertIn("效用", html)
        self.assertIn("(3, 4)", html)
        self.assertNotIn("confidence_delta_c", html)
        self.assertNotIn("low_confidence_high_risk_path_ratio", html)
        self.assertIn("invalid", html)
        self.assertIn("obstacle", html)

        self.assertTrue(report.summary["path_reachable"])
        self.assertGreater(report.summary["path_nodes"], 1)
        self.assertGreaterEqual(report.summary["hard_constraint_violations"], 0)
        self.assertEqual(report.summary["image_path"], str(report.image_path))
        self.assertEqual(report.summary["html_path"], str(report.html_path))

    def test_visualize_minimal_closure_script_writes_outputs_and_json_summary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "visualize_minimal_closure.py"

        output_dir = Path(tempfile.mkdtemp(prefix="dev-platform-constraints-visualization-cli-"))
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
                "--confidence-config",
                str(repo_root / "configs" / "confidence" / "default.json"),
                "--width",
                "16",
                "--height",
                "10",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        image_path = Path(summary["image_path"])
        html_path = Path(summary["html_path"])

        self.assertTrue(image_path.exists())
        self.assertTrue(html_path.exists())
        self.assertEqual(image_path.name, "minimal_closure.png")
        self.assertEqual(html_path.name, "minimal_closure.html")
        self.assertTrue(summary["path_reachable"])
        self.assertGreater(summary["path_nodes"], 1)
        self.assertTrue(np.isfinite(float(summary["path_total_cost"])))
        self.assertGreaterEqual(summary["hard_constraint_violations"], 0)
        self.assertIn("confidence_mean_before", summary)
        self.assertIn("confidence_mean_after", summary)
        self.assertIn("confidence_delta_c", summary)
        self.assertIn("coverage_rate", summary)
        self.assertIn("coverage_rate_delta", summary)
        self.assertIn("low_confidence_high_risk_path_ratio", summary)
        self.assertIn("data_contract", summary)
        self.assertIn("top_exploration_goals", summary)
        self.assertEqual(summary["data_contract"]["issue_summary"]["errors"], 0)
        self.assertGreater(len(summary["top_exploration_goals"]), 0)
        self.assertGreater(float(summary["confidence_delta_c"]), 0.0)

        html = html_path.read_text(encoding="utf-8")
        self.assertIn("可信度更新摘要", html)
        self.assertIn("覆盖率摘要", html)
        self.assertIn("数据契约摘要", html)
        self.assertIn("探索目标摘要", html)
        self.assertNotIn("confidence_delta_c", html)


if __name__ == "__main__":
    unittest.main()
