import unittest

from dev_platform_constraints.exploration import CandidateGoal, ExplorationWeights, rank_exploration_goals


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


if __name__ == "__main__":
    unittest.main()
