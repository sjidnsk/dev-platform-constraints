import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev_platform_constraints.confidence import (
    BayesianStateLayer,
    ConfidenceWeights,
    derive_confidence_from_posterior,
    load_confidence_weights,
    update_obstacle_posterior,
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


if __name__ == "__main__":
    unittest.main()
