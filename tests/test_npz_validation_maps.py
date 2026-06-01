import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev_platform_constraints.experiments import load_ablation_scenarios


class NpzValidationMapGenerationTests(unittest.TestCase):
    def test_generator_dry_run_prints_planned_scenarios_without_writing_maps(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = Path(tempfile.mkdtemp(prefix="npz-validation-dry-run-")) / "maps"
        script = repo_root / "scripts" / "generate_npz_validation_maps.py"

        result = subprocess.run(
            [sys.executable, str(script), "--output-dir", str(output_dir), "--dry-run"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["output_dir"], str(output_dir))
        self.assertGreaterEqual(len(summary["scenarios"]), 3)
        self.assertFalse(output_dir.exists())

    def test_generator_writes_reproducible_npz_maps_and_scenario_config(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "generate_npz_validation_maps.py"
        root_a = Path(tempfile.mkdtemp(prefix="npz-validation-a-"))
        root_b = Path(tempfile.mkdtemp(prefix="npz-validation-b-"))

        for root in (root_a, root_b):
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output-dir",
                    str(root / "maps"),
                    "--scenario-config",
                    str(root / "npz_validation_scenarios.json"),
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        scenarios_a = load_ablation_scenarios(root_a / "npz_validation_scenarios.json")
        scenarios_b = load_ablation_scenarios(root_b / "npz_validation_scenarios.json")
        self.assertEqual([item.scenario_id for item in scenarios_a], [item.scenario_id for item in scenarios_b])
        self.assertTrue(all(item.map_source.kind == "npz_grid" for item in scenarios_a))

        first_a = Path(scenarios_a[0].map_source.path)
        first_b = Path(scenarios_b[0].map_source.path)
        with np.load(first_a, allow_pickle=False) as grid_a, np.load(first_b, allow_pickle=False) as grid_b:
            self.assertEqual(set(grid_a.files), set(grid_b.files))
            for name in ("elevation", "obstacle", "illumination", "confidence", "value", "valid_mask"):
                with self.subTest(layer=name):
                    self.assertTrue(np.array_equal(grid_a[name], grid_b[name]))

    def test_tracked_npz_validation_scenario_config_points_to_generated_maps(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scenarios = load_ablation_scenarios(repo_root / "configs" / "ablation" / "npz_validation_scenarios.json")

        self.assertGreaterEqual(len(scenarios), 3)
        self.assertEqual({scenario.map_source.kind for scenario in scenarios}, {"npz_grid"})
        self.assertTrue(all("data" in str(scenario.map_source.path) for scenario in scenarios))

    def test_path_planner_sidecar_export_dry_run_does_not_write_files(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = Path(tempfile.mkdtemp(prefix="path-sidecar-dry-run-")) / "exports"
        script = repo_root / "scripts" / "export_path_planner_sidecars.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--scenario-config",
                str(repo_root / "configs" / "ablation" / "npz_validation_scenarios.json"),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(len(summary["exports"]), 3)
        self.assertFalse(output_dir.exists())

    def test_path_planner_sidecar_export_writes_three_npz_contract_pairs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="path-sidecar-export-"))
        generator = repo_root / "scripts" / "generate_npz_validation_maps.py"
        exporter = repo_root / "scripts" / "export_path_planner_sidecars.py"
        scenario_config = root / "npz_validation_scenarios.json"

        generated = subprocess.run(
            [
                sys.executable,
                str(generator),
                "--output-dir",
                str(root / "maps"),
                "--scenario-config",
                str(scenario_config),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)

        exported = subprocess.run(
            [
                sys.executable,
                str(exporter),
                "--scenario-config",
                str(scenario_config),
                "--output-dir",
                str(root / "exports"),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)
        summary = json.loads(exported.stdout)
        scenario_ids = {item["scenario_id"] for item in summary["exports"]}
        self.assertEqual(
            scenario_ids,
            {"npz_shadow_corridor", "npz_rock_field_multi_pose", "npz_low_confidence_risk_band"},
        )
        self.assertTrue((root / "exports" / "manifest.json").exists())
        for item in summary["exports"]:
            contract = json.loads(Path(item["contract"]).read_text(encoding="utf-8"))
            sidecar = json.loads(Path(item["sidecar"]).read_text(encoding="utf-8"))
            self.assertEqual(contract["schema_version"], "model-explorer-contract/v1")
            self.assertNotIn("cost", contract)
            self.assertNotIn("passable_mask", contract)
            self.assertEqual(sidecar["schema_version"], "path-planner-sidecar/v1")
            self.assertIn("cost", sidecar)
            self.assertIn("passable_mask", sidecar)
            self.assertEqual(sidecar["metadata"]["scenario_id"], item["scenario_id"])


if __name__ == "__main__":
    unittest.main()
