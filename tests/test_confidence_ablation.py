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
        self.assertEqual(len(summary["runs"]), 3)
        self.assertTrue(Path(summary["json_path"]).exists())
        self.assertTrue(Path(summary["csv_path"]).exists())
        self.assertTrue(Path(summary["html_path"]).exists())
        self.assertTrue(Path(summary["image_path"]).exists())
        self.assertEqual(Path(summary["image_path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        first_run = summary["runs"][0]
        self.assertIn("confidence_config", first_run)
        self.assertIn("path_total_cost", first_run)
        self.assertIn("confidence_delta_c", first_run)
        self.assertIn("low_confidence_high_risk_path_ratio", first_run)
        self.assertIn("top_goal_cells", first_run)
        self.assertLessEqual(len(first_run["top_goal_cells"]), 2)

        with Path(summary["csv_path"]).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertIn("top_goal_cells", rows[0])

        html = Path(summary["html_path"]).read_text(encoding="utf-8")
        self.assertIn("可信度权重消融报告", html)
        self.assertIn("路径总代价", html)
        self.assertIn("可信度正向提升总量", html)
        self.assertIn("Top-K 探索目标", html)
        self.assertIn("confidence_ablation.png", html)


if __name__ == "__main__":
    unittest.main()
