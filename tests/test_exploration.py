import unittest

from dev_platform_constraints.exploration import (
    CandidateGoal,
    ExplorationWeights,
    GoalSequenceEvaluation,
    evaluate_goal_sequences,
    generate_exploration_candidates,
    rank_exploration_goals,
)
from dev_platform_constraints.confidence import update_coverage_from_observation
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.path_planning import astar_path
from dev_platform_constraints.platforms import ParameterValue, PlatformParameters
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


class ExplorationGoalTests(unittest.TestCase):
    def sensor_platform(self, sensor_range: float, sensor_fov: float) -> PlatformParameters:
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        parameters = dict(platform.parameters)
        parameters["sensor_range"] = ParameterValue(sensor_range, "m", "assumed", "测试观测距离")
        parameters["sensor_fov"] = ParameterValue(sensor_fov, "deg", "assumed", "测试视场角")
        return PlatformParameters(name=platform.name, parameters=parameters)

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

    def test_generate_candidates_uses_sensor_footprint_for_expected_confidence_gain(self) -> None:
        grid = generate_sample_grid(width=10, height=7, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["confidence"][:] = 0.9
        grid.layers["value"][:] = 0.0
        grid.layers["confidence"][3, 7] = 0.1
        grid.layers["value"][3, 7] = 1.0
        platform = self.sensor_platform(sensor_range=4.0, sensor_fov=70.0)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        candidates = generate_exploration_candidates(grid, constraints, start=(0, 3), platform=platform, max_candidates=10)

        forward_candidate = next(candidate for candidate in candidates if candidate.cell == (4, 3))
        self.assertGreater(forward_candidate.confidence_gain, 0.1)
        self.assertGreater(forward_candidate.information_gain, 0.1)

    def test_two_step_lookahead_adds_downstream_footprint_gain_without_changing_reachability(self) -> None:
        grid = generate_sample_grid(width=12, height=7, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["confidence"][:] = 0.9
        grid.layers["value"][:] = 0.0
        grid.layers["confidence"][3, 4] = 0.6
        grid.layers["confidence"][3, 7] = 0.1
        grid.layers["value"][3, 7] = 1.0
        platform = self.sensor_platform(sensor_range=3.0, sensor_fov=70.0)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        single_step = generate_exploration_candidates(
            grid,
            constraints,
            start=(0, 3),
            platform=platform,
            max_candidates=12,
            lookahead_steps=1,
        )
        two_step = generate_exploration_candidates(
            grid,
            constraints,
            start=(0, 3),
            platform=platform,
            max_candidates=12,
            lookahead_steps=2,
        )

        single_candidate = next(candidate for candidate in single_step if candidate.cell == (4, 3))
        two_step_candidate = next(candidate for candidate in two_step if candidate.cell == (4, 3))
        self.assertTrue(two_step_candidate.reachable)
        self.assertEqual(single_candidate.reachable, two_step_candidate.reachable)
        self.assertGreater(two_step_candidate.confidence_gain, single_candidate.confidence_gain)
        self.assertGreater(two_step_candidate.information_gain, single_candidate.information_gain)

    def test_simple_occlusion_reduces_hidden_footprint_gain_and_changes_candidate_order(self) -> None:
        grid = generate_sample_grid(width=10, height=7, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["obstacle"][:] = 0.0
        grid.layers["obstacle_height"][:] = 0.0
        grid.layers["confidence"][:] = 0.9
        grid.layers["value"][:] = 0.0
        grid.layers["obstacle"][3, 5] = 1.0
        grid.layers["confidence"][3, 7] = 0.1
        grid.layers["value"][3, 7] = 1.0
        grid.layers["confidence"][5, 4] = 0.3
        grid.layers["value"][5, 4] = 0.6
        platform = self.sensor_platform(sensor_range=5.0, sensor_fov=75.0)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        open_candidates = generate_exploration_candidates(
            grid,
            constraints,
            start=(0, 3),
            platform=platform,
            max_candidates=50,
            use_simple_occlusion=False,
        )
        occluded_candidates = generate_exploration_candidates(
            grid,
            constraints,
            start=(0, 3),
            platform=platform,
            max_candidates=50,
            use_simple_occlusion=True,
        )

        open_target = next(candidate for candidate in open_candidates if candidate.cell == (3, 3))
        occluded_target = next(candidate for candidate in occluded_candidates if candidate.cell == (3, 3))
        open_order = [scored.candidate.cell for scored in rank_exploration_goals(open_candidates)]
        occluded_order = [scored.candidate.cell for scored in rank_exploration_goals(occluded_candidates)]
        self.assertGreater(open_target.confidence_gain, occluded_target.confidence_gain)
        self.assertLess(open_order.index((3, 3)), occluded_order.index((3, 3)))

    def test_generate_candidates_reports_expected_new_coverage_after_existing_coverage(self) -> None:
        grid = generate_sample_grid(width=10, height=7, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["confidence"][:] = 0.9
        grid.layers["value"][:] = 0.0
        grid.layers["confidence"][3, 4] = 0.1
        grid.layers["value"][3, 4] = 1.0
        grid.layers["confidence"][3, 7] = 0.1
        grid.layers["value"][3, 7] = 1.0
        platform = self.sensor_platform(sensor_range=4.0, sensor_fov=70.0)
        update_coverage_from_observation(grid, platform, observer_cell=(4, 3), heading_deg=0.0)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        candidates = generate_exploration_candidates(
            grid,
            constraints,
            start=(0, 3),
            platform=platform,
            max_candidates=50,
        )

        covered_target = next(candidate for candidate in candidates if candidate.cell == (4, 3))
        fresh_target = next(candidate for candidate in candidates if candidate.cell == (7, 3))
        self.assertGreater(covered_target.coverage_area, 0.0)
        self.assertGreater(len(covered_target.coverage_cells), 0)
        self.assertLess(covered_target.expected_new_coverage_area, covered_target.coverage_area)
        self.assertGreater(fresh_target.expected_new_coverage_area, covered_target.expected_new_coverage_area)
        self.assertGreater(fresh_target.expected_coverage_rate_delta, covered_target.expected_coverage_rate_delta)

    def test_generate_candidates_uses_new_coverage_to_seed_top_candidates(self) -> None:
        grid = generate_sample_grid(width=12, height=7, resolution=1.0)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        grid.layers["confidence"][:] = 0.5
        grid.layers["value"][:] = 0.0
        platform = self.sensor_platform(sensor_range=5.0, sensor_fov=90.0)
        update_coverage_from_observation(grid, platform, observer_cell=(2, 3), heading_deg=0.0)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)

        candidates = generate_exploration_candidates(
            grid,
            constraints,
            start=(0, 3),
            platform=platform,
            max_candidates=5,
        )

        self.assertTrue(candidates)
        self.assertTrue(any(candidate.expected_new_coverage_area > 0.0 for candidate in candidates))
        self.assertTrue(all(candidate.coverage_cells for candidate in candidates))

    def test_evaluate_goal_sequences_penalizes_unreachable_and_high_risk_sequences(self) -> None:
        goals = (
            CandidateGoal(cell=(2, 2), information_gain=0.7, value=0.7, confidence_gain=0.7, risk=0.1, path_cost=2.0),
            CandidateGoal(cell=(4, 2), information_gain=1.0, value=1.0, confidence_gain=1.0, risk=0.95, path_cost=2.0),
            CandidateGoal(
                cell=(6, 2),
                information_gain=1.0,
                value=1.0,
                confidence_gain=1.0,
                risk=0.1,
                path_cost=1.0,
                reachable=False,
            ),
        )

        evaluations = evaluate_goal_sequences(goals, depth=2, beam_width=3)

        self.assertTrue(all(isinstance(item, GoalSequenceEvaluation) for item in evaluations))
        self.assertEqual(evaluations[0].goals[0].cell, (2, 2))
        self.assertTrue(evaluations[0].reachable)
        self.assertGreater(evaluations[0].delta_c, 0.0)
        self.assertGreater(evaluations[0].value_coverage, 0.0)
        self.assertGreater(evaluations[0].coverage_area, 0.0)
        self.assertEqual(len(evaluations[0].segment_path_costs), len(evaluations[0].goals))
        self.assertGreaterEqual(evaluations[0].cumulative_risk, evaluations[0].risk)
        self.assertTrue(all(not item.reachable or item.utility > -1.0 for item in evaluations))
        unreachable = next(item for item in evaluations if any(goal.cell == (6, 2) for goal in item.goals))
        self.assertIn("unreachable:(6, 2)", unreachable.unreachable_reasons)
        self.assertLess(
            max(item.utility for item in evaluations if any(goal.cell == (6, 2) for goal in item.goals)),
            evaluations[0].utility,
        )
        self.assertLess(
            max(item.utility for item in evaluations if any(goal.cell == (4, 2) for goal in item.goals)),
            evaluations[0].utility,
        )

    def test_evaluate_goal_sequences_deduplicates_coverage_and_explains_risk(self) -> None:
        goals = (
            CandidateGoal(
                cell=(1, 1),
                information_gain=0.8,
                value=0.8,
                confidence_gain=0.8,
                risk=0.1,
                path_cost=1.0,
                coverage_area=2.0,
                coverage_cells=((1, 1), (2, 1)),
            ),
            CandidateGoal(
                cell=(2, 1),
                information_gain=0.7,
                value=0.7,
                confidence_gain=0.7,
                risk=0.2,
                path_cost=1.2,
                coverage_area=2.0,
                coverage_cells=((2, 1), (3, 1)),
            ),
            CandidateGoal(
                cell=(3, 1),
                information_gain=1.0,
                value=1.0,
                confidence_gain=1.0,
                risk=0.9,
                path_cost=1.1,
                coverage_area=2.0,
                coverage_cells=((4, 1), (5, 1)),
            ),
        )

        evaluations = evaluate_goal_sequences(goals, depth=2, beam_width=3)

        overlapping = next(
            item
            for item in evaluations
            if tuple(goal.cell for goal in item.goals) == ((1, 1), (2, 1))
        )
        self.assertEqual(overlapping.coverage_area, 3.0)
        self.assertEqual(overlapping.risk_reasons, tuple())
        high_risk = next(item for item in evaluations if any(goal.cell == (3, 1) for goal in item.goals))
        self.assertIn("high_risk:(3, 1):0.900", high_risk.risk_reasons)
        self.assertLess(
            max(item.utility for item in evaluations if any(goal.cell == (3, 1) for goal in item.goals)),
            overlapping.utility,
        )


if __name__ == "__main__":
    unittest.main()
