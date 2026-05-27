import unittest
from importlib import import_module


class PackageStructureTests(unittest.TestCase):
    def test_canonical_subpackages_export_existing_public_api(self) -> None:
        from dev_platform_constraints.confidence import (
            BayesianStateLayer,
            ConfidenceComponent,
            ConfidenceUpdateReport,
            ConfidenceWeights,
            ObservationModelResult,
            TerrainLikelihoodRules,
            compute_terrain_category_likelihood,
            compute_observation_confidence,
            compute_observation_model,
            default_terrain_likelihood_config_path,
            derive_confidence_from_posterior,
            fuse_confidence,
            load_confidence_weights,
            load_terrain_likelihood_rules,
            update_obstacle_posterior,
            update_traversability_posterior,
            update_confidence_from_observation,
        )
        from dev_platform_constraints.core import CORE_LAYERS, GridMap, LayerMetadata, validate_grid_map
        from dev_platform_constraints.mapping import CostWeights, ReasonCode, generate_costmap, generate_hard_constraints
        from dev_platform_constraints.path_planning import astar_path
        from dev_platform_constraints.platforms import PlatformParameters, load_platform_parameters
        from dev_platform_constraints.reporting import (
            MODEL_EXPLORER_SCHEMA_VERSION,
            MODEL_EXPLORER_STABLE_FIELDS,
            build_data_contract_report,
            build_model_explorer_contract,
            render_closure_report,
        )
        from dev_platform_constraints.sample_data import generate_sample_grid, load_npz_grid
        from dev_platform_constraints.terrain import derive_terrain_features
        from dev_platform_constraints.exploration import CandidateGoal, evaluate_goal_sequences, generate_exploration_candidates, rank_exploration_goals
        from dev_platform_constraints.experiments import AblationScenario, load_ablation_scenarios

        self.assertTrue(CORE_LAYERS)
        self.assertTrue(callable(validate_grid_map))
        self.assertTrue(callable(generate_hard_constraints))
        self.assertTrue(callable(generate_costmap))
        self.assertTrue(callable(astar_path))
        self.assertTrue(callable(load_platform_parameters))
        self.assertTrue(callable(render_closure_report))
        self.assertTrue(callable(generate_sample_grid))
        self.assertTrue(callable(load_npz_grid))
        self.assertTrue(callable(derive_terrain_features))
        self.assertTrue(callable(compute_terrain_category_likelihood))
        self.assertTrue(callable(default_terrain_likelihood_config_path))
        self.assertTrue(callable(load_terrain_likelihood_rules))
        self.assertTrue(callable(compute_observation_confidence))
        self.assertTrue(callable(compute_observation_model))
        self.assertTrue(callable(fuse_confidence))
        self.assertTrue(callable(load_confidence_weights))
        self.assertTrue(callable(update_obstacle_posterior))
        self.assertTrue(callable(update_traversability_posterior))
        self.assertTrue(callable(derive_confidence_from_posterior))
        self.assertTrue(callable(update_confidence_from_observation))
        self.assertTrue(callable(build_data_contract_report))
        self.assertTrue(callable(build_model_explorer_contract))
        self.assertEqual(MODEL_EXPLORER_SCHEMA_VERSION, "model-explorer-contract/v1")
        self.assertIn("top_sequences.coverage_area", MODEL_EXPLORER_STABLE_FIELDS)
        self.assertTrue(callable(generate_exploration_candidates))
        self.assertTrue(callable(evaluate_goal_sequences))
        self.assertTrue(callable(rank_exploration_goals))
        self.assertTrue(callable(load_ablation_scenarios))
        self.assertEqual(GridMap.__name__, "GridMap")
        self.assertEqual(LayerMetadata.__name__, "LayerMetadata")
        self.assertEqual(CostWeights.__name__, "CostWeights")
        self.assertEqual(PlatformParameters.__name__, "PlatformParameters")
        self.assertEqual(ConfidenceComponent.__name__, "ConfidenceComponent")
        self.assertEqual(ConfidenceWeights.__name__, "ConfidenceWeights")
        self.assertEqual(TerrainLikelihoodRules.__name__, "TerrainLikelihoodRules")
        self.assertEqual(ConfidenceUpdateReport.__name__, "ConfidenceUpdateReport")
        self.assertEqual(ObservationModelResult.__name__, "ObservationModelResult")
        self.assertEqual(BayesianStateLayer.__name__, "BayesianStateLayer")
        self.assertEqual(CandidateGoal.__name__, "CandidateGoal")
        self.assertEqual(AblationScenario.__name__, "AblationScenario")
        self.assertTrue(hasattr(ReasonCode, "SLOPE"))

    def test_legacy_flat_modules_are_removed(self) -> None:
        for module_name in (
            "dev_platform_constraints.constraints",
            "dev_platform_constraints.costmap",
            "dev_platform_constraints.data_contracts",
            "dev_platform_constraints.examples",
            "dev_platform_constraints.map_layers",
            "dev_platform_constraints.planning",
            "dev_platform_constraints.platform_model",
            "dev_platform_constraints.terrain_features",
            "dev_platform_constraints.visualization",
        ):
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    import_module(module_name)


if __name__ == "__main__":
    unittest.main()
