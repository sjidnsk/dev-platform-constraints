import unittest

import numpy as np

from dev_platform_constraints.mapping import ReasonCode, generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import PlatformParameters
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


class ConstraintCostmapPlanningTests(unittest.TestCase):
    def test_hard_constraints_report_slope_obstacle_forbidden_and_invalid_reasons(self) -> None:
        grid = generate_sample_grid(width=6, height=5, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=1.0)
        grid.layers["slope"][0, 0] = 30.0
        grid.layers["obstacle"][0, 1] = 1.0
        grid.layers["forbidden"] = np.zeros((5, 6), dtype=float)
        grid.layers["forbidden"][0, 2] = 1.0
        grid.layers["valid_mask"][0, 3] = False

        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        result = generate_hard_constraints(grid, platform)

        self.assertFalse(bool(result.passable_mask[0, 0]))
        self.assertTrue(result.reason_layer[0, 0] & ReasonCode.SLOPE)
        self.assertTrue(result.reason_layer[0, 1] & ReasonCode.OBSTACLE)
        self.assertTrue(result.reason_layer[0, 2] & ReasonCode.FORBIDDEN)
        self.assertTrue(result.reason_layer[0, 3] & ReasonCode.INVALID)

    def test_costmap_is_nonnegative_and_keeps_traversability_separate_from_confidence(self) -> None:
        grid = generate_sample_grid(width=8, height=6, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=1.0)
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)

        grid.layers["confidence"][:] = 0.2
        generate_costmap(grid, constraints, platform)

        self.assertGreaterEqual(float(np.nanmin(grid.layers["cost"])), 0.0)
        self.assertTrue(np.all((grid.layers["traversability"] >= 0.0) & (grid.layers["traversability"] <= 1.0)))
        self.assertFalse(np.allclose(grid.layers["traversability"], grid.layers["confidence"]))

    def test_astar_finds_path_through_minimal_closure_outputs(self) -> None:
        grid = generate_sample_grid(width=16, height=10, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        result = astar_path(grid.layers["cost"], constraints.passable_mask, start=(0, 0), goal=(15, 9), resolution=grid.resolution)

        self.assertTrue(result.reachable)
        self.assertGreater(len(result.path), 1)
        self.assertGreaterEqual(result.total_cost, 0.0)
        for x, y in result.path:
            self.assertTrue(bool(constraints.passable_mask[y, x]))


if __name__ == "__main__":
    unittest.main()
