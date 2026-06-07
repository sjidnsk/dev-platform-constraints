import json
import unittest
from pathlib import Path

import numpy as np

from dev_platform_constraints.core import GridMap, metadata_for_generated_layer
from dev_platform_constraints.exploration import evaluate_goal_sequences, generate_exploration_candidates, rank_exploration_goals
from dev_platform_constraints.mapping import ConstraintResult, generate_costmap, generate_hard_constraints
from dev_platform_constraints.platforms import ParameterValue, PlatformParameters
from dev_platform_constraints.reporting import (
    MODEL_EXPLORER_SCHEMA_VERSION,
    MODEL_EXPLORER_STABLE_FIELDS,
    build_model_explorer_contract,
    build_path_planner_sidecar,
)
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


class ModelExplorerContractTests(unittest.TestCase):
    def test_contract_contains_stable_model_explorer_sections(self) -> None:
        grid = generate_sample_grid(width=12, height=8, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)
        candidates = generate_exploration_candidates(grid, constraints, start=(0, 0), platform=platform, max_candidates=5)
        scored_goals = rank_exploration_goals(candidates)[:3]
        sequences = evaluate_goal_sequences(candidates, depth=2, beam_width=3)[:2]
        confidence_report = {
            "mean_confidence_before": 0.7,
            "mean_confidence_after": 0.75,
            "delta_c": 1.2,
            "visible_cell_count": 5,
            "updated_cell_count": 4,
        }

        contract = build_model_explorer_contract(grid, constraints, scored_goals, sequences, confidence_report)

        self.assertEqual(contract["schema_version"], MODEL_EXPLORER_SCHEMA_VERSION)
        self.assertEqual(contract["grid"]["width"], 12)
        self.assertIn("reason_counts", contract["constraints"])
        self.assertIn("top_goals", contract)
        self.assertIn("top_sequences", contract)
        self.assertIn("observation_update", contract)
        self.assertIn("stable_fields", contract)
        self.assertIn("experimental_fields", contract)
        self.assertIn("cell", contract["top_goals"][0])
        self.assertIn("coverage_area", contract["top_goals"][0])
        self.assertIn("expected_new_coverage_area", contract["top_goals"][0])
        self.assertIn("expected_coverage_rate_delta", contract["top_goals"][0])
        self.assertIn("energy_cost", contract["top_goals"][0])
        self.assertIn("coverage_area", contract["top_sequences"][0])
        self.assertIn("segment_path_costs", contract["top_sequences"][0])
        self.assertEqual(contract["stable_fields"], list(MODEL_EXPLORER_STABLE_FIELDS))
        self.assertIn("top_goals.expected_new_coverage_area", contract["experimental_fields"])
        self.assertIn("top_goals.expected_coverage_rate_delta", contract["experimental_fields"])
        self.assertIn("top_goals.energy_cost", contract["experimental_fields"])

    def test_path_planner_sidecar_contains_cost_and_passable_mask(self) -> None:
        grid = generate_sample_grid(width=8, height=6, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        sidecar = build_path_planner_sidecar(
            grid,
            constraints,
            scenario_id="unit-sidecar",
            map_source={"kind": "sample_grid"},
            platform="test-rover",
        )

        self.assertEqual(sidecar["schema_version"], "path-planner-sidecar/v1")
        self.assertEqual(sidecar["grid"]["width"], 8)
        self.assertEqual(len(sidecar["cost"]), 6)
        self.assertEqual(len(sidecar["cost"][0]), 8)
        self.assertEqual(len(sidecar["passable_mask"]), 6)
        self.assertIn("confidence", sidecar["terrain_layers"])
        self.assertEqual(sidecar["metadata"]["scenario_id"], "unit-sidecar")
        cost = np.asarray(sidecar["cost"], dtype=float)
        passable_mask = np.asarray(sidecar["passable_mask"])
        self.assertEqual(cost.shape, (6, 8))
        self.assertEqual(passable_mask.shape, (6, 8))
        self.assertEqual(passable_mask.dtype, np.dtype("bool"))
        self.assertTrue(np.all(cost >= 0.0))
        self.assertTrue(np.array_equal(passable_mask, constraints.passable_mask))
        self.assertTrue(np.all(~passable_mask[constraints.reason_layer != 0]))

    def test_path_planner_sidecar_metadata_exposes_platform_goal_admissibility(self) -> None:
        grid = GridMap(resolution=1.0, origin=(0.0, 0.0), width=5, height=5, frame_id="moon_local")
        grid.add_layer(
            "cost",
            np.ones((5, 5), dtype=float),
            metadata_for_generated_layer("cost", resolution=1.0, frame_id="moon_local"),
        )
        passable_mask = np.ones((5, 5), dtype=bool)
        passable_mask[2, 2] = False
        reason_layer = np.zeros((5, 5), dtype=np.uint16)
        reason_layer[2, 2] = 4
        constraints = ConstraintResult(
            passable_mask=passable_mask,
            reason_layer=reason_layer,
            reason_names={4: "obstacle"},
        )
        platform = PlatformParameters(
            name="footprint-test-rover",
            parameters={
                "max_slope_deg": ParameterValue(20.0, "deg", "assumed", "unit test"),
                "max_obstacle_height": ParameterValue(0.2, "m", "assumed", "unit test"),
                "ground_clearance": ParameterValue(0.18, "m", "assumed", "unit test"),
                "min_turning_radius": ParameterValue(0.0, "m", "assumed", "unit test"),
                "sensor_range": ParameterValue(5.0, "m", "assumed", "unit test"),
                "sensor_fov": ParameterValue(60.0, "deg", "assumed", "unit test"),
                "energy_model": ParameterValue({"base_cost": 1.0}, "relative", "assumed", "unit test"),
                "body_length": ParameterValue(2.0, "m", "estimated", "unit test"),
                "body_width": ParameterValue(2.0, "m", "estimated", "unit test"),
            },
        )

        sidecar = build_path_planner_sidecar(
            grid,
            constraints,
            scenario_id="unit-platform-goal-admissibility",
            platform="footprint-test-rover",
            platform_parameters=platform,
            include_terrain_layers=False,
        )

        admissibility = sidecar["metadata"]["platform_goal_admissibility"]
        self.assertEqual(admissibility["schema_version"], "platform-goal-admissibility/v1")
        self.assertEqual(admissibility["passable_source"], "inflated_passable_mask")
        self.assertGreater(admissibility["footprint_radius_m"], 1.0)
        self.assertEqual(admissibility["original_blocked_count"], 1)
        self.assertGreater(admissibility["inflated_blocked_count"], admissibility["original_blocked_count"])
        self.assertFalse(admissibility["inflated_passable_mask"][2][3])
        self.assertTrue(admissibility["inflated_passable_mask"][0][0])
        self.assertEqual(admissibility["cell_roles"]["policy_target_cell"], "model_explorer_contract_top_goal")
        self.assertEqual(admissibility["cell_roles"]["execution_goal_cell"], "same_cell_when_inflated_passable")
        self.assertEqual(
            admissibility["cell_roles"]["nearest_inflated_passable_anchor"],
            "audit_projection_candidate_when_policy_target_is_not_inflated_passable",
        )
        self.assertEqual(admissibility["training_use"]["audit_proxy_anchor_not_same_cell"], "not_positive_evidence")

    def test_documented_contract_example_matches_stable_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        example_path = repo_root / "docs" / "model-explorer-contract-example.json"

        example = json.loads(example_path.read_text(encoding="utf-8"))

        self.assertEqual(example["schema_version"], "model-explorer-contract/v1")
        for key in ("grid", "constraints", "top_goals", "top_sequences", "observation_update"):
            with self.subTest(key=key):
                self.assertIn(key, example)

    def test_minimal_contract_example_uses_only_stable_required_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        example_path = repo_root / "docs" / "model-explorer-minimal-example.json"

        example = json.loads(example_path.read_text(encoding="utf-8"))

        self.assertEqual(example["schema_version"], MODEL_EXPLORER_SCHEMA_VERSION)
        self.assertEqual(example["stable_fields"], list(MODEL_EXPLORER_STABLE_FIELDS))
        self.assertEqual(sorted(example), sorted(("schema_version", "grid", "constraints", "top_goals", "top_sequences", "observation_update", "stable_fields", "experimental_fields")))


if __name__ == "__main__":
    unittest.main()
