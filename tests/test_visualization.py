import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev_platform_constraints.constraints import generate_hard_constraints
from dev_platform_constraints.costmap import generate_costmap
from dev_platform_constraints.examples import generate_sample_grid
from dev_platform_constraints.planning import astar_path
from dev_platform_constraints.platform_model import PlatformParameters
from dev_platform_constraints.terrain_features import derive_terrain_features
from dev_platform_constraints.visualization import render_closure_report


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
        report = render_closure_report(grid, constraints, plan, output_dir, title="测试可视化")

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


if __name__ == "__main__":
    unittest.main()
