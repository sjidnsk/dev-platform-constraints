import unittest

from dev_platform_constraints.exploration import CandidateGoal, ExplorationWeights, generate_exploration_candidates, rank_exploration_goals
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import PlatformParameters
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


class ExplorationGoalTests(unittest.TestCase):
    def test_low_confidence_high_value_reachable_goal_ranks_first(self) -> None:
        goals = (
            CandidateGoal(cell=(1, 1), information_gain=0.2, value=0.2, confidence_gain=0.1, risk=0.1, path_cost=1.0),
            CandidateGoal(cell=(2, 2), information_gain=0.6, value=0.9, confidence_gain=0.8, risk=0.2, path_cost=3.0),
        )

        ranked = rank_exploration_goals(goals)

        self.assertEqual(ranked[0].candidate.cell, (2, 2))
        self.assertGreater(ranked[0].utility, ranked[1].utility)

    def test_unreachable_goal_is_penalized_even_when_raw_gain_is_high(self) -> None:
        goals = (
            CandidateGoal(cell=(1, 1), information_gain=0.4, value=0.4, confidence_gain=0.4, risk=0.2, path_cost=2.0),
            CandidateGoal(
                cell=(9, 9),
                information_gain=1.0,
                value=1.0,
                confidence_gain=1.0,
                risk=0.1,
                path_cost=1.0,
                reachable=False,
            ),
        )

        ranked = rank_exploration_goals(goals)

        self.assertEqual(ranked[0].candidate.cell, (1, 1))
        self.assertLess(ranked[1].utility, ranked[0].utility)

    def test_normalization_prevents_large_path_cost_scale_from_dominating(self) -> None:
        goals = (
            CandidateGoal(cell=(1, 1), information_gain=1.0, value=1.0, confidence_gain=1.0, risk=0.1, path_cost=100.0),
            CandidateGoal(cell=(2, 2), information_gain=0.1, value=0.1, confidence_gain=0.1, risk=0.1, path_cost=1.0),
        )
        weights = ExplorationWeights(path_cost=0.1)

        ranked = rank_exploration_goals(goals, weights=weights)

        self.assertEqual(ranked[0].candidate.cell, (1, 1))
        self.assertLessEqual(ranked[0].normalized_terms["path_cost"], 1.0)

    def test_generate_candidates_prefers_low_confidence_high_value_cells_and_limits_count(self) -> None:
        grid = generate_sample_grid(width=12, height=8, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["confidence"][:] = 0.8
        grid.layers["value"][:] = 0.0
        grid.layers["confidence"][5, 9] = 0.2
        grid.layers["value"][5, 9] = 1.0
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        candidates = generate_exploration_candidates(grid, constraints, start=(0, 0), platform=platform, max_candidates=3)

        self.assertLessEqual(len(candidates), 3)
        self.assertIn((9, 5), {candidate.cell for candidate in candidates})
        target = next(candidate for candidate in candidates if candidate.cell == (9, 5))
        self.assertGreater(target.value, 0.0)
        self.assertGreater(target.confidence_gain, 0.0)
        self.assertTrue(target.reachable)
        self.assertGreater(target.path_cost, 0.0)

    def test_generate_candidates_marks_unreachable_cells_for_sorting_penalty(self) -> None:
        grid = generate_sample_grid(width=8, height=6, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["confidence"][:] = 0.9
        grid.layers["value"][:] = 0.0
        grid.layers["confidence"][4, 6] = 0.1
        grid.layers["value"][4, 6] = 1.0
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        constraints = generate_hard_constraints(grid, platform)
        passable = constraints.passable_mask.copy()
        passable[:, 4] = False
        constraints = type(constraints)(passable_mask=passable, reason_layer=constraints.reason_layer, reason_names=constraints.reason_names)
        generate_costmap(grid, constraints, platform)

        candidates = generate_exploration_candidates(grid, constraints, start=(0, 0), platform=platform, max_candidates=5)

        target = next(candidate for candidate in candidates if candidate.cell == (6, 4))
        self.assertFalse(target.reachable)
        self.assertEqual(astar_path(grid.layers["cost"], constraints.passable_mask, (0, 0), target.cell, grid.resolution).path, tuple())


if __name__ == "__main__":
    unittest.main()
