import json
import unittest
from pathlib import Path

from dev_platform_constraints.exploration import evaluate_goal_sequences, generate_exploration_candidates, rank_exploration_goals
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.platforms import PlatformParameters
from dev_platform_constraints.reporting import build_model_explorer_contract
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

        self.assertEqual(contract["schema_version"], "model-explorer-contract/v1")
        self.assertEqual(contract["grid"]["width"], 12)
        self.assertIn("reason_counts", contract["constraints"])
        self.assertIn("top_goals", contract)
        self.assertIn("top_sequences", contract)
        self.assertIn("observation_update", contract)
        self.assertIn("stable_fields", contract)
        self.assertIn("experimental_fields", contract)
        self.assertIn("cell", contract["top_goals"][0])
        self.assertIn("coverage_area", contract["top_sequences"][0])
        self.assertIn("segment_path_costs", contract["top_sequences"][0])

    def test_documented_contract_example_matches_stable_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        example_path = repo_root / "docs" / "model-explorer-contract-example.json"

        example = json.loads(example_path.read_text(encoding="utf-8"))

        self.assertEqual(example["schema_version"], "model-explorer-contract/v1")
        for key in ("grid", "constraints", "top_goals", "top_sequences", "observation_update"):
            with self.subTest(key=key):
                self.assertIn(key, example)


if __name__ == "__main__":
    unittest.main()
