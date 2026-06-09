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

    def test_generator_can_emit_stress_scenarios_with_blocked_regions(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "generate_npz_validation_maps.py"
        root = Path(tempfile.mkdtemp(prefix="npz-validation-stress-"))
        scenario_config = root / "npz_validation_scenarios.json"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--scenario-set",
                "stress",
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

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        scenario_ids = {item["scenario_id"] for item in summary["scenarios"]}
        self.assertIn("npz_mixed_stress_detour", scenario_ids)
        self.assertIn("npz_path_complexity_benefit_probe", scenario_ids)
        self.assertIn("npz_low_centerline_bad_channel", scenario_ids)
        self.assertIn("npz_blocked_nearby_clearance_detour", scenario_ids)
        self.assertIn("npz_high_cost_exposure_rock_detour", scenario_ids)
        scenarios = load_ablation_scenarios(scenario_config)
        self.assertEqual(len(scenarios), 8)
        for scenario in scenarios:
            with np.load(Path(scenario.map_source.path), allow_pickle=False) as grid:
                self.assertEqual(grid["obstacle"].shape, (scenario.height, scenario.width))
                if scenario.scenario_id != "npz_low_centerline_bad_channel":
                    self.assertGreater(np.count_nonzero(grid["obstacle"] >= 0.5), 0)
                self.assertLess(float(np.min(grid["confidence"])), 0.5)
                self.assertTrue(np.any(grid["value"] > 0.0))

        scenario_entries = json.loads(scenario_config.read_text(encoding="utf-8"))["scenarios"]
        contrast_entries = {
            item["scenario_id"]: item for item in scenario_entries if item.get("scenario_group") == "channel_contrast"
        }
        self.assertEqual(
            {
                "npz_low_centerline_bad_channel",
                "npz_blocked_nearby_clearance_detour",
                "npz_high_cost_exposure_rock_detour",
            },
            set(contrast_entries),
        )
        self.assertEqual(
            contrast_entries["npz_low_centerline_bad_channel"]["contrast_focus"],
            "low_centerline_cost_bad_channel_quality",
        )
        self.assertEqual(
            contrast_entries["npz_blocked_nearby_clearance_detour"]["contrast_focus"],
            "blocked_nearby_clearance",
        )
        self.assertEqual(
            contrast_entries["npz_high_cost_exposure_rock_detour"]["contrast_focus"],
            "high_cost_exposure_rock_field_detour",
        )
        probe_entry = next(
            item for item in scenario_entries if item["scenario_id"] == "npz_path_complexity_benefit_probe"
        )
        self.assertEqual(probe_entry["scenario_group"], "stress")
        self.assertEqual(probe_entry["risk_region"], [8, 20, 4, 14])
        self.assertGreaterEqual(len(probe_entry["blocked_rects"]), 4)
        with np.load(Path(probe_entry["map_source"]["path"]), allow_pickle=False) as grid:
            high_risk_band = grid["confidence"][4:14, 8:20]
            upper_corridor = grid["confidence"][1:4, 8:20]
            lower_corridor = grid["confidence"][15:18, 8:20]
            self.assertLess(float(np.max(high_risk_band)), 0.35)
            self.assertGreater(float(np.mean(upper_corridor)), float(np.mean(high_risk_band)))
            self.assertGreater(float(np.mean(lower_corridor)), float(np.mean(high_risk_band)))

    def test_mixed_stress_export_has_reachable_and_blocked_candidates(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="npz-validation-mixed-"))
        scenario_config = root / "npz_validation_scenarios.json"

        generated = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "generate_npz_validation_maps.py"),
                "--scenario-set",
                "stress",
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
                str(repo_root / "scripts" / "export_path_planner_sidecars.py"),
                "--scenario-config",
                str(scenario_config),
                "--output-dir",
                str(root / "exports"),
                "--top-k",
                "6",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)

        scenarios = json.loads(scenario_config.read_text(encoding="utf-8"))["scenarios"]
        mixed = [item for item in scenarios if item["scenario_group"] == "mixed_stress"]
        self.assertEqual([item["scenario_id"] for item in mixed], ["npz_mixed_stress_detour"])
        contract = json.loads((root / "exports" / "npz_mixed_stress_detour.contract.json").read_text(encoding="utf-8"))
        sidecar = json.loads(
            (root / "exports" / "npz_mixed_stress_detour.path-planner-sidecar.json").read_text(encoding="utf-8")
        )
        reachable_values = {bool(goal["reachable"]) for goal in contract["top_goals"]}
        self.assertEqual(reachable_values, {False, True})
        self.assertEqual(len(sidecar["cost"]), contract["grid"]["height"])
        self.assertEqual(len(sidecar["passable_mask"][0]), contract["grid"]["width"])

    def test_policy_canary_value_stability_covers_six_families_with_unique_variants(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="npz-canary-value-"))
        scenario_config = root / "npz_validation_scenarios.json"

        generated = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "generate_npz_validation_maps.py"),
                "--scenario-set",
                "policy_canary_value_stability",
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
        scenarios = json.loads(scenario_config.read_text(encoding="utf-8"))["scenarios"]
        self.assertEqual(len(scenarios), 36)
        families = {}
        scenario_ids = set()
        seeds = set()
        variants = set()
        geometry_signatures = set()
        for scenario in scenarios:
            families[scenario["scenario_group"]] = families.get(scenario["scenario_group"], 0) + 1
            scenario_ids.add(scenario["scenario_id"])
            seeds.add(scenario["seed"])
            variants.add(scenario["scenario_variant_id"])
            geometry_signatures.add(
                (
                    tuple(scenario.get("risk_region", [])),
                    tuple(scenario.get("value_region", [])),
                    tuple(tuple(rect) for rect in scenario.get("blocked_rects", [])),
                )
            )
        self.assertEqual(
            families,
            {
                "mixed_stress_detour": 6,
                "near_blocked_safe_alt": 6,
                "high_risk_tradeoff": 6,
                "dense_choke_safe_bypass": 6,
                "channel_contrast": 6,
                "path_complexity_benefit": 6,
            },
        )
        self.assertEqual(len(scenario_ids), 36)
        self.assertEqual(len(seeds), 36)
        self.assertEqual(len(variants), 36)
        self.assertGreaterEqual(len(geometry_signatures), 12)

    def test_explicit_scenario_spec_overrides_start_cell_and_identity(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="npz-explicit-spec-"))
        scenario_spec = root / "explicit-scenario-spec.json"
        scenario_config = root / "npz_validation_scenarios.json"
        scenario_spec.write_text(
            json.dumps(
                {
                    "schema_version": "npz-validation-explicit-scenario-spec/v1",
                    "scenario_set": "policy_canary_value_stability",
                    "scenarios": [
                        {
                            "template_scenario_id": "npz_canary_value_stability_mixed_stress_detour_a",
                            "scenario_id": "npz_seq_canary_mixed_stress_detour_ep00_step01",
                            "scenario_group": "mixed_stress_detour",
                            "scenario_seed": 12001,
                            "scenario_variant_id": "seq-mixed-ep00-step01",
                            "start_cell": [7, 6],
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        generated = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "generate_npz_validation_maps.py"),
                "--scenario-spec-json",
                str(scenario_spec),
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
        summary = json.loads(generated.stdout)
        self.assertEqual(summary["scenario_set"], "explicit")
        self.assertEqual(summary["scenarios"][0]["scenario_id"], "npz_seq_canary_mixed_stress_detour_ep00_step01")
        scenario = json.loads(scenario_config.read_text(encoding="utf-8"))["scenarios"][0]
        self.assertEqual(scenario["start_cell"], [7, 6])
        self.assertEqual(scenario["scenario_group"], "mixed_stress_detour")
        self.assertEqual(scenario["seed"], 12001)
        self.assertEqual(scenario["scenario_variant_id"], "seq-mixed-ep00-step01")
        self.assertTrue((root / "maps" / "npz_seq_canary_mixed_stress_detour_ep00_step01.npz").is_file())

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
