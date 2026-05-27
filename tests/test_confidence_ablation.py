import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ConfidenceAblationScriptTests(unittest.TestCase):
    def test_ablation_script_outputs_json_csv_and_goal_ranking_metrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = Path(tempfile.mkdtemp(prefix="confidence-ablation-"))
        script = repo_root / "scripts" / "run_confidence_ablation.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
                "--configs",
                str(repo_root / "configs" / "confidence" / "default.json"),
                str(repo_root / "configs" / "confidence" / "observation_focused.json"),
                str(repo_root / "configs" / "confidence" / "consistency_recency_focused.json"),
                "--scenario-config",
                str(repo_root / "configs" / "ablation" / "scenarios.json"),
                "--top-k",
                "2",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertGreater(len(summary["scenarios"]), 1)
        self.assertEqual(len(summary["runs"]), len(summary["scenarios"]) * 3)
        self.assertIn("aggregate", summary)
        self.assertIn("recommendation", summary)
        self.assertEqual(summary["recommendation"]["confidence_config"], "consistency_recency_focused.json")
        self.assertIn("推荐", summary["recommendation"]["reason"])
        self.assertIn("risk_conflict_gap", {scenario["scenario_id"] for scenario in summary["scenarios"]})
        self.assertTrue(any(scenario["map_source"]["kind"] == "seeded_synthetic" for scenario in summary["scenarios"]))
        self.assertGreater(summary["recommendation"]["risk_conflict_hit_rate"], 0.0)
        self.assertTrue(
            any(
                run["scenario_id"] == "risk_conflict_gap" and run["low_confidence_high_risk_path_ratio"] > 0.0
                for run in summary["runs"]
            )
        )
        self.assertTrue(Path(summary["json_path"]).exists())
        self.assertTrue(Path(summary["csv_path"]).exists())
        self.assertTrue(Path(summary["html_path"]).exists())
        self.assertTrue(Path(summary["image_path"]).exists())
        self.assertEqual(Path(summary["image_path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        first_run = summary["runs"][0]
        self.assertIn("scenario_id", first_run)
        self.assertIn("confidence_config", first_run)
        self.assertIn("path_total_cost", first_run)
        self.assertIn("confidence_delta_c", first_run)
        self.assertIn("low_confidence_high_risk_path_ratio", first_run)
        self.assertIn("top_goal_cells", first_run)
        self.assertIn("top_goal_sequence_cells", first_run)
        self.assertIn("top_goal_sequence_utilities", first_run)
        self.assertIn("top_goal_sequence_details", first_run)
        self.assertIn("terrain_model_confidence_mean", first_run)
        self.assertIn("map_source_kind", first_run)
        self.assertIn("coverage_area", first_run["top_goal_sequence_details"][0])
        self.assertIn("segment_path_costs", first_run["top_goal_sequence_details"][0])
        self.assertIn("unreachable_reasons", first_run["top_goal_sequence_details"][0])
        self.assertLessEqual(len(first_run["top_goal_cells"]), 2)
        self.assertGreater(len(summary["runs"]), len(summary["aggregate"]))

        first_aggregate = summary["aggregate"][0]
        self.assertIn("path_total_cost_mean", first_aggregate)
        self.assertIn("path_total_cost_std", first_aggregate)
        self.assertIn("confidence_delta_c_mean", first_aggregate)
        self.assertIn("best_path_cost_count", first_aggregate)
        self.assertIn("top_goal_stability", first_aggregate)
        self.assertIn("top_goal_stability_change", first_aggregate)
        self.assertIn("risk_conflict_hit_rate", first_aggregate)
        self.assertIn("failure_scenarios", first_aggregate)
        self.assertIn("map_family_best_path_cost_count", first_aggregate)
        self.assertIn("sequence_goal_stability", first_aggregate)

        with Path(summary["csv_path"]).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(summary["runs"]))
        self.assertIn("top_goal_cells", rows[0])

        html = Path(summary["html_path"]).read_text(encoding="utf-8")
        self.assertIn("可信度权重消融报告", html)
        self.assertIn("路径总代价", html)
        self.assertIn("可信度正向提升总量", html)
        self.assertIn("Top-K 探索目标", html)
        self.assertIn("配置胜率", html)
        self.assertIn("Top-K 稳定性", html)
        self.assertIn("序列目标稳定性", html)
        self.assertIn("推荐配置", html)
        self.assertIn("风险冲突命中率", html)
        self.assertIn("confidence_ablation.png", html)

    def test_ablation_script_accepts_external_scenario_config(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = Path(tempfile.mkdtemp(prefix="confidence-ablation-custom-"))
        scenario_path = output_dir / "scenario.json"
        scenario_path.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "scenario_id": "custom_small",
                            "width": 16,
                            "height": 10,
                            "resolution": 0.5,
                            "observations": [{"observer_cell": [0, 5], "heading_deg": 0.0}],
                            "start_cell": [0, 0],
                            "goal_cell": [15, 9],
                            "elapsed_time": 1.0,
                            "recency_time_constant": 10.0,
                            "low_confidence_band": [4, 6],
                            "value_region": [12, 16, 7, 10],
                            "map_source": {"kind": "sample"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        script = repo_root / "scripts" / "run_confidence_ablation.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
                "--configs",
                str(repo_root / "configs" / "confidence" / "default.json"),
                "--scenario-config",
                str(scenario_path),
                "--top-k",
                "1",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual([scenario["scenario_id"] for scenario in summary["scenarios"]], ["custom_small"])
        self.assertEqual(len(summary["runs"]), 1)
        self.assertEqual(summary["runs"][0]["scenario_id"], "custom_small")
        self.assertEqual(summary["runs"][0]["map_source_kind"], "sample")

    def test_ablation_script_accepts_npz_grid_scenario(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = Path(tempfile.mkdtemp(prefix="confidence-ablation-npz-"))
        grid_path = output_dir / "external_grid.npz"
        shape = (10, 16)
        zeros = np.zeros(shape, dtype=float)
        np.savez(
            grid_path,
            resolution=0.5,
            elevation=np.tile(np.linspace(0.0, 0.2, shape[1]), (shape[0], 1)),
            obstacle=zeros,
            obstacle_height=zeros,
            illumination=np.full(shape, 0.8, dtype=float),
            confidence=np.full(shape, 0.7, dtype=float),
            value=zeros,
            valid_mask=np.ones(shape, dtype=bool),
        )
        scenario_path = output_dir / "scenario.json"
        scenario_path.write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "scenario_id": "npz_external",
                            "width": 16,
                            "height": 10,
                            "resolution": 0.5,
                            "observations": [{"observer_cell": [0, 5], "heading_deg": 0.0}],
                            "start_cell": [0, 0],
                            "goal_cell": [15, 9],
                            "elapsed_time": 1.0,
                            "recency_time_constant": 10.0,
                            "low_confidence_band": [4, 6],
                            "value_region": [12, 16, 7, 10],
                            "map_source": {"kind": "npz_grid", "path": str(grid_path)},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        script = repo_root / "scripts" / "run_confidence_ablation.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
                "--configs",
                str(repo_root / "configs" / "confidence" / "default.json"),
                "--scenario-config",
                str(scenario_path),
                "--top-k",
                "2",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["runs"][0]["map_source_kind"], "npz_grid")
        self.assertTrue(summary["runs"][0]["validation_valid"])
        self.assertTrue(summary["runs"][0]["path_reachable"])
        self.assertGreater(summary["runs"][0]["terrain_model_confidence_mean"], 0.0)

    def test_ablation_script_accepts_terrain_likelihood_config(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = Path(tempfile.mkdtemp(prefix="confidence-ablation-terrain-rules-"))
        scenario_path = output_dir / "scenario.json"
        rules_path = output_dir / "terrain_likelihood.json"
        scenario_path.write_text(
            json.dumps(
                {
                    "terrain_likelihood_config": str(rules_path),
                    "scenarios": [
                        {
                            "scenario_id": "custom_terrain_rules",
                            "width": 16,
                            "height": 10,
                            "resolution": 0.5,
                            "observations": [{"observer_cell": [0, 5], "heading_deg": 0.0}],
                            "start_cell": [0, 0],
                            "goal_cell": [15, 9],
                            "elapsed_time": 1.0,
                            "recency_time_constant": 10.0,
                            "low_confidence_band": [4, 6],
                            "value_region": [12, 16, 7, 10],
                            "map_source": {"kind": "sample"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        rules_path.write_text(
            json.dumps(
                {
                    "rules": {
                        "slope_rough_deg": 12.0,
                        "roughness_threshold": 0.4,
                        "obstacle_threshold": 0.45,
                        "shadow_threshold": 0.45,
                        "base_likelihood": 0.05,
                        "risk_likelihood": 0.95,
                    }
                }
            ),
            encoding="utf-8",
        )
        script = repo_root / "scripts" / "run_confidence_ablation.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
                "--configs",
                str(repo_root / "configs" / "confidence" / "default.json"),
                "--scenario-config",
                str(scenario_path),
                "--top-k",
                "1",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["terrain_likelihood_config"], str(rules_path))
        self.assertEqual(summary["runs"][0]["terrain_likelihood_config"], rules_path.name)
        self.assertGreater(summary["runs"][0]["terrain_model_confidence_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
