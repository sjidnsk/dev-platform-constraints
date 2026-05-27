import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
