import json
import tempfile
import unittest
from pathlib import Path

from dev_platform_constraints.experiments import (
    AblationScenario,
    default_ablation_scenario_config_path,
    load_ablation_scenarios,
)


class AblationScenarioConfigTests(unittest.TestCase):
    def test_default_scenario_config_loads_extended_deterministic_scenarios(self) -> None:
        scenarios = load_ablation_scenarios(default_ablation_scenario_config_path())
        scenario_ids = {scenario.scenario_id for scenario in scenarios}

        self.assertGreaterEqual(len(scenarios), 6)
        self.assertIn("baseline_gap", scenario_ids)
        self.assertIn("risk_conflict_gap", scenario_ids)
        self.assertIn("simple_occlusion", scenario_ids)
        self.assertIn("multi_observation", scenario_ids)
        self.assertTrue(all(isinstance(scenario, AblationScenario) for scenario in scenarios))
        self.assertTrue(next(scenario for scenario in scenarios if scenario.scenario_id == "simple_occlusion").use_simple_occlusion)
        self.assertGreater(len(next(scenario for scenario in scenarios if scenario.scenario_id == "multi_observation").observations), 1)

    def test_invalid_scenario_config_rejects_bad_dimensions_and_out_of_bounds_cells(self) -> None:
        invalid = {
            "scenarios": [
                {
                    "scenario_id": "bad",
                    "width": 0,
                    "height": 4,
                    "resolution": 0.5,
                    "observations": [{"observer_cell": [5, 5], "heading_deg": 0.0}],
                    "start_cell": [0, 0],
                    "goal_cell": [6, 6],
                    "elapsed_time": 1.0,
                    "recency_time_constant": 10.0,
                    "low_confidence_band": [1, 2],
                    "value_region": [2, 3, 2, 3],
                }
            ]
        }
        path = Path(tempfile.mkdtemp(prefix="ablation-scenario-")) / "bad.json"
        path.write_text(json.dumps(invalid), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "width and height"):
            load_ablation_scenarios(path)


if __name__ == "__main__":
    unittest.main()
