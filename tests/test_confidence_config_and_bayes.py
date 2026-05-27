import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev_platform_constraints.confidence import (
    BayesianStateLayer,
    CategoricalBayesianStateLayer,
    ConfidenceWeights,
    TerrainLikelihoodRules,
    compute_terrain_category_likelihood,
    derive_confidence_from_categorical_posterior,
    derive_confidence_from_posterior,
    load_confidence_weights,
    update_categorical_posterior,
    update_obstacle_posterior,
    update_traversability_posterior,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfidenceConfigAndBayesTests(unittest.TestCase):
    def test_default_confidence_config_matches_dataclass_defaults(self) -> None:
        weights = load_confidence_weights(REPO_ROOT / "configs" / "confidence" / "default.json")

        self.assertEqual(weights, ConfidenceWeights())

    def test_confidence_config_rejects_negative_and_all_zero_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="confidence-config-") as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text(
                json.dumps(
                    {
                        "weights": {
                            "resolution": -0.1,
                            "observation": 0.0,
                            "recency": 0.0,
                            "consistency": 0.0,
                            "model": 0.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "nonnegative"):
                load_confidence_weights(config_path)

            config_path.write_text(
                json.dumps(
                    {
                        "weights": {
                            "resolution": 0.0,
                            "observation": 0.0,
                            "recency": 0.0,
                            "consistency": 0.0,
                            "model": 0.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "positive"):
                load_confidence_weights(config_path)

    def test_obstacle_posterior_moves_with_high_quality_observations_and_skips_invalid_cells(self) -> None:
        prior = np.array([[0.2, 0.8], [0.9, 0.4]])
        observed = np.array([[0.9, 0.1], [0.1, 0.8]])
        quality = np.array([[0.9, 0.9], [1.0, 0.9]])
        valid = np.array([[True, True], [True, False]])

        posterior = update_obstacle_posterior(prior, observed, quality, valid_mask=valid)

        self.assertEqual(posterior.name, "obstacle")
        self.assertGreater(float(posterior.values[0, 0]), float(prior[0, 0]))
        self.assertLess(float(posterior.values[0, 1]), float(prior[0, 1]))
        self.assertAlmostEqual(float(posterior.values[1, 0]), 0.5)
        self.assertFalse(bool(posterior.valid_mask[1, 1]))
        self.assertAlmostEqual(float(posterior.values[1, 1]), float(prior[1, 1]))

    def test_confidence_derived_from_posterior_is_lowest_for_ambiguous_probability(self) -> None:
        posterior = BayesianStateLayer(
            name="obstacle",
            values=np.array([[0.01, 0.5, 0.99]]),
            valid_mask=np.ones((1, 3), dtype=bool),
        )

        confidence = derive_confidence_from_posterior(posterior)

        self.assertEqual(confidence.name, "model")
        self.assertGreater(float(confidence.values[0, 0]), float(confidence.values[0, 1]))
        self.assertGreater(float(confidence.values[0, 2]), float(confidence.values[0, 1]))
        self.assertAlmostEqual(float(confidence.values[0, 1]), 0.0)

    def test_traversability_posterior_moves_with_observations_and_skips_invalid_cells(self) -> None:
        prior = np.array([[0.3, 0.7], [0.2, 0.8]])
        observed = np.array([[0.9, 0.1], [0.8, 0.2]])
        quality = np.array([[0.9, 0.9], [1.0, 1.0]])
        valid = np.array([[True, True], [True, False]])

        posterior = update_traversability_posterior(prior, observed, quality, valid_mask=valid)

        self.assertEqual(posterior.name, "traversable")
        self.assertGreater(float(posterior.values[0, 0]), float(prior[0, 0]))
        self.assertLess(float(posterior.values[0, 1]), float(prior[0, 1]))
        self.assertGreater(float(posterior.values[1, 0]), float(prior[1, 0]))
        self.assertFalse(bool(posterior.valid_mask[1, 1]))
        self.assertAlmostEqual(float(posterior.values[1, 1]), float(prior[1, 1]))

    def test_conflicting_traversability_observation_lowers_derived_confidence(self) -> None:
        prior = np.full((1, 2), 0.9)
        agreeing = update_traversability_posterior(prior, np.array([[0.9, 0.9]]), np.ones((1, 2)))
        conflicting = update_traversability_posterior(prior, np.array([[0.1, 0.1]]), np.ones((1, 2)))

        agreeing_confidence = derive_confidence_from_posterior(agreeing)
        conflicting_confidence = derive_confidence_from_posterior(conflicting)

        self.assertLess(float(conflicting_confidence.values[0, 0]), float(agreeing_confidence.values[0, 0]))

    def test_categorical_posterior_moves_toward_high_quality_terrain_observation_and_skips_invalid_cells(self) -> None:
        categories = ("safe_regolith", "rough", "obstacle", "shadow_risk")
        prior = np.full((2, 2, 4), 0.25)
        likelihood = np.full((2, 2, 4), 0.05)
        likelihood[0, 0] = np.array([0.05, 0.85, 0.05, 0.05])
        likelihood[0, 1] = np.array([0.05, 0.05, 0.85, 0.05])
        likelihood[1, 0] = np.array([0.05, 0.05, 0.05, 0.85])
        likelihood[1, 1] = np.array([0.85, 0.05, 0.05, 0.05])
        quality = np.array([[1.0, 0.8], [0.6, 1.0]])
        valid = np.array([[True, True], [True, False]])

        posterior = update_categorical_posterior(
            prior,
            likelihood,
            quality,
            categories=categories,
            valid_mask=valid,
        )

        self.assertIsInstance(posterior, CategoricalBayesianStateLayer)
        self.assertEqual(posterior.categories, categories)
        self.assertEqual(posterior.name, "terrain")
        self.assertGreater(float(posterior.probabilities[0, 0, 1]), 0.7)
        self.assertGreater(float(posterior.probabilities[0, 1, 2]), float(prior[0, 1, 2]))
        self.assertGreater(float(posterior.probabilities[1, 0, 3]), float(prior[1, 0, 3]))
        self.assertFalse(bool(posterior.valid_mask[1, 1]))
        self.assertTrue(np.allclose(posterior.probabilities[1, 1], prior[1, 1]))

    def test_categorical_conflict_lowers_entropy_derived_confidence(self) -> None:
        categories = ("safe_regolith", "rough", "obstacle", "shadow_risk")
        confident_prior = np.array([[[0.85, 0.05, 0.05, 0.05]]])
        agreeing_likelihood = np.array([[[0.90, 0.04, 0.03, 0.03]]])
        conflicting_likelihood = np.array([[[0.05, 0.85, 0.05, 0.05]]])

        agreeing = update_categorical_posterior(
            confident_prior,
            agreeing_likelihood,
            np.ones((1, 1)),
            categories=categories,
        )
        conflicting = update_categorical_posterior(
            confident_prior,
            conflicting_likelihood,
            np.ones((1, 1)),
            categories=categories,
        )

        agreeing_confidence = derive_confidence_from_categorical_posterior(agreeing)
        conflicting_confidence = derive_confidence_from_categorical_posterior(conflicting)

        self.assertEqual(agreeing_confidence.name, "model")
        self.assertLess(float(conflicting_confidence.values[0, 0]), float(agreeing_confidence.values[0, 0]))

    def test_terrain_category_likelihood_uses_risk_layers_and_configurable_rules(self) -> None:
        slope = np.array([[2.0, 18.0], [5.0, 3.0]])
        roughness = np.array([[0.1, 0.9], [0.2, 0.1]])
        obstacle = np.array([[0.0, 0.1], [0.9, 0.1]])
        illumination = np.array([[0.8, 0.8], [0.7, 0.1]])

        likelihood = compute_terrain_category_likelihood(
            slope=slope,
            roughness=roughness,
            obstacle=obstacle,
            illumination=illumination,
        )

        category_index = {name: index for index, name in enumerate(likelihood.categories)}
        self.assertGreater(likelihood.probabilities[0, 0, category_index["safe_regolith"]], 0.6)
        self.assertGreater(likelihood.probabilities[0, 1, category_index["rough"]], 0.6)
        self.assertGreater(likelihood.probabilities[1, 0, category_index["obstacle"]], 0.6)
        self.assertGreater(likelihood.probabilities[1, 1, category_index["shadow_risk"]], 0.6)

        stricter_rules = TerrainLikelihoodRules(roughness_threshold=0.95)
        stricter = compute_terrain_category_likelihood(
            slope=slope,
            roughness=roughness,
            obstacle=obstacle,
            illumination=illumination,
            rules=stricter_rules,
        )
        self.assertLess(
            stricter.probabilities[0, 1, category_index["rough"]],
            likelihood.probabilities[0, 1, category_index["rough"]],
        )


if __name__ == "__main__":
    unittest.main()
