import json
import subprocess
import sys
import unittest
from pathlib import Path

from dev_platform_constraints.core import validate_grid_map
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.platforms import PlatformParameters
from dev_platform_constraints.reporting import build_data_contract_report
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


class DataContractReportTests(unittest.TestCase):
    def test_data_contract_report_contains_layer_metadata_and_issue_summary(self) -> None:
        grid = generate_sample_grid(width=8, height=6, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)
        validation = validate_grid_map(grid)

        report = build_data_contract_report(grid, validation)

        self.assertTrue(report["is_valid"])
        self.assertIn("elevation", report["layers"])
        self.assertIn("cost", report["layers"])
        self.assertEqual(report["layers"]["elevation"]["unit"], "m")
        self.assertEqual(report["layers"]["elevation"]["resolution"], 0.5)
        self.assertIn("valid_ratio", report["layers"]["confidence"])
        self.assertEqual(report["issue_summary"], {"errors": 0, "warnings": 0})

    def test_run_minimal_closure_outputs_data_contract_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = Path("scripts") / "run_minimal_closure.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)

        self.assertIn("data_contract", summary)
        self.assertIn("coverage_rate", summary)
        self.assertIn("coverage_rate_delta", summary)
        self.assertGreater(summary["covered_valid_cell_count"], 0)
        self.assertTrue(summary["data_contract"]["is_valid"])
        self.assertIn("confidence", summary["data_contract"]["layers"])
        self.assertIn("coverage_mask", summary["data_contract"]["layers"])
        self.assertEqual(summary["data_contract"]["issue_summary"]["errors"], 0)


if __name__ == "__main__":
    unittest.main()
