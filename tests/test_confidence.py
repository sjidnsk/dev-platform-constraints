import unittest

import numpy as np

from dev_platform_constraints.confidence import (
    ConfidenceComponent,
    ConfidenceWeights,
    ObservationModelResult,
    compute_consistency_confidence,
    compute_observation_confidence,
    compute_observation_model,
    compute_recency_confidence,
    compute_resolution_confidence,
    fuse_confidence,
    update_confidence_from_observation,
)
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.platforms import ParameterValue, PlatformParameters
from dev_platform_constraints.sample_data import generate_sample_grid
from dev_platform_constraints.terrain import derive_terrain_features


class ConfidenceTests(unittest.TestCase):
    def sensor_platform(self, sensor_range: float, sensor_fov: float) -> PlatformParameters:
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        parameters = dict(platform.parameters)
        parameters["sensor_range"] = ParameterValue(sensor_range, "m", "assumed", "测试观测距离")
        parameters["sensor_fov"] = ParameterValue(sensor_fov, "deg", "assumed", "测试视场角")
        return PlatformParameters(name=platform.name, parameters=parameters)

    def test_resolution_confidence_decreases_when_grid_is_coarser_than_planning_scale(self) -> None:
        fine = compute_resolution_confidence(0.5, planning_resolution=1.0, shape=(2, 3))
        coarse = compute_resolution_confidence(2.0, planning_resolution=1.0, shape=(2, 3))

        self.assertEqual(fine.name, "resolution")
        self.assertTrue(np.allclose(fine.values, 1.0))
        self.assertTrue(np.allclose(coarse.values, 0.5))
        self.assertTrue(np.all(coarse.valid_mask))

    def test_observation_confidence_uses_sensor_range_and_fov(self) -> None:
        grid = generate_sample_grid(width=9, height=5, resolution=1.0)
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)

        observed = compute_observation_confidence(grid, platform, observer_cell=(0, 2), heading_deg=0.0)

        self.assertEqual(observed.name, "observation")
        self.assertGreater(float(observed.values[2, 1]), 0.0)
        self.assertGreater(float(observed.values[2, 1]), float(observed.values[2, 5]))
        self.assertEqual(float(observed.values[4, 1]), 0.0)
        self.assertTrue(bool(observed.valid_mask[2, 1]))
        self.assertFalse(bool(observed.valid_mask[4, 1]))

    def test_observation_model_returns_quality_layer_and_component(self) -> None:
        grid = generate_sample_grid(width=6, height=5, resolution=1.0)
        platform = self.sensor_platform(sensor_range=4.0, sensor_fov=90.0)

        model = compute_observation_model(grid, platform, observer_cell=(1, 2), heading_deg=0.0)

        self.assertIsInstance(model, ObservationModelResult)
        self.assertTrue(bool(model.visible_mask[2, 3]))
        self.assertFalse(bool(model.visible_mask[4, 1]))
        self.assertGreater(float(model.quality_layer[2, 2]), float(model.quality_layer[2, 5]))
        self.assertEqual(model.confidence_component.name, "observation")
        self.assertGreater(float(model.confidence_component.values[2, 3]), 0.0)

    def test_observation_model_can_lower_quality_behind_simple_obstacles(self) -> None:
        grid = generate_sample_grid(width=7, height=5, resolution=1.0)
        grid.layers["obstacle"][:] = 0.0
        grid.layers["obstacle"][2, 3] = 1.0
        platform = self.sensor_platform(sensor_range=6.0, sensor_fov=60.0)

        open_model = compute_observation_model(grid, platform, observer_cell=(1, 2), heading_deg=0.0)
        occluded_model = compute_observation_model(grid, platform, observer_cell=(1, 2), heading_deg=0.0, use_simple_occlusion=True)

        self.assertGreater(float(open_model.quality_layer[2, 5]), float(occluded_model.quality_layer[2, 5]))
        self.assertEqual(float(occluded_model.quality_layer[2, 5]), 0.0)

    def test_fuse_confidence_renormalizes_missing_components_and_marks_all_missing_invalid(self) -> None:
        valid = np.array([[True, False], [True, True]])
        resolution = ConfidenceComponent("resolution", np.full((2, 2), 1.0), np.ones((2, 2), dtype=bool))
        observation = ConfidenceComponent("observation", np.full((2, 2), 0.2), valid)

        fused, fused_valid = fuse_confidence(
            (resolution, observation),
            weights=ConfidenceWeights(resolution=0.25, observation=0.75, recency=0.0, consistency=0.0, model=0.0),
        )

        self.assertAlmostEqual(float(fused[0, 0]), 0.4)
        self.assertAlmostEqual(float(fused[0, 1]), 1.0)
        self.assertTrue(bool(fused_valid[0, 1]))

        missing, missing_valid = fuse_confidence(
            (ConfidenceComponent("observation", np.zeros((2, 2)), np.zeros((2, 2), dtype=bool)),)
        )
        self.assertTrue(np.allclose(missing, 0.0))
        self.assertFalse(np.any(missing_valid))

    def test_recency_and_consistency_components_are_bounded_and_explainable(self) -> None:
        recency = compute_recency_confidence((2, 2), elapsed_time=5.0, time_constant=10.0)
        consistency = compute_consistency_confidence(
            np.array([[0.0, 1.0], [0.0, np.nan]]),
            np.array([[0.0, 2.0], [0.0, 1.0]]),
            sigma=1.0,
        )

        self.assertTrue(np.all((recency.values > 0.0) & (recency.values <= 1.0)))
        self.assertAlmostEqual(float(consistency.values[0, 0]), 1.0)
        self.assertLess(float(consistency.values[0, 1]), 1.0)
        self.assertFalse(bool(consistency.valid_mask[1, 1]))

    def test_observation_update_raises_visible_confidence_and_decays_unobserved_cells(self) -> None:
        grid = generate_sample_grid(width=16, height=10, resolution=0.5)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        before = grid.layers["confidence"].copy()
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)

        report = update_confidence_from_observation(
            grid,
            platform,
            observer_cell=(0, grid.height // 2),
            heading_deg=0.0,
            elapsed_time=2.0,
            recency_time_constant=10.0,
        )

        after = grid.layers["confidence"]
        self.assertGreater(report.visible_cell_count, 0)
        self.assertGreater(report.delta_c, 0.0)
        self.assertAlmostEqual(
            report.mean_confidence_delta,
            report.mean_confidence_after - report.mean_confidence_before,
        )
        self.assertGreater(float(after[grid.height // 2, 1]), float(before[grid.height // 2, 1]))
        self.assertLess(float(after[0, grid.width - 1]), float(before[0, grid.width - 1]))
        self.assertLessEqual(report.low_confidence_area_after, report.low_confidence_area_before)

    def test_low_confidence_high_risk_cells_increase_cost_without_becoming_hard_constraints(self) -> None:
        high_risk = generate_sample_grid(width=4, height=4, resolution=1.0)
        low_risk = generate_sample_grid(width=4, height=4, resolution=1.0)
        for grid in (high_risk, low_risk):
            derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=1.0)
            grid.layers["confidence"][:] = 1.0
            grid.layers["confidence"][1, 1] = 0.1
            grid.layers["obstacle"][:] = 0.0
            grid.layers["obstacle_height"][:] = 0.0
            grid.layers["illumination"][:] = 1.0
        high_risk.layers["roughness"][1, 1] = 1.0
        low_risk.layers["roughness"][1, 1] = 0.0

        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        high_constraints = generate_hard_constraints(high_risk, platform)
        low_constraints = generate_hard_constraints(low_risk, platform)
        generate_costmap(high_risk, high_constraints, platform)
        generate_costmap(low_risk, low_constraints, platform)

        self.assertTrue(bool(high_constraints.passable_mask[1, 1]))
        self.assertTrue(bool(low_constraints.passable_mask[1, 1]))
        self.assertGreater(float(high_risk.layers["cost"][1, 1]), float(low_risk.layers["cost"][1, 1]))


if __name__ == "__main__":
    unittest.main()
