import unittest

import numpy as np

from dev_platform_constraints.confidence import (
    CoverageEstimate,
    CoverageUpdateReport,
    ensure_coverage_mask,
    estimate_observation_coverage,
    update_coverage_from_observation,
)
from dev_platform_constraints.platforms import ParameterValue, PlatformParameters
from dev_platform_constraints.sample_data import generate_sample_grid


class CoverageTests(unittest.TestCase):
    def sensor_platform(self, sensor_range: float, sensor_fov: float) -> PlatformParameters:
        platform = PlatformParameters.minimal(name="test-rover", max_slope_deg=20.0, max_obstacle_height=0.2)
        parameters = dict(platform.parameters)
        parameters["sensor_range"] = ParameterValue(sensor_range, "m", "assumed", "测试观测距离")
        parameters["sensor_fov"] = ParameterValue(sensor_fov, "deg", "assumed", "测试视场角")
        return PlatformParameters(name=platform.name, parameters=parameters)

    def test_ensure_coverage_mask_initializes_missing_layer_as_uncovered(self) -> None:
        grid = generate_sample_grid(width=5, height=4, resolution=0.5)

        coverage = ensure_coverage_mask(grid)

        self.assertEqual(coverage.shape, grid.shape)
        self.assertEqual(coverage.dtype, np.bool_)
        self.assertFalse(np.any(coverage))
        self.assertIn("coverage_mask", grid.layers)
        self.assertEqual(grid.metadata["coverage_mask"].unit, "boolean")

    def test_observation_update_reports_unique_valid_map_coverage(self) -> None:
        grid = generate_sample_grid(width=8, height=5, resolution=1.0)
        platform = self.sensor_platform(sensor_range=6.0, sensor_fov=70.0)

        first = update_coverage_from_observation(grid, platform, observer_cell=(1, 2), heading_deg=0.0)
        second = update_coverage_from_observation(grid, platform, observer_cell=(1, 2), heading_deg=0.0)

        self.assertIsInstance(first, CoverageUpdateReport)
        self.assertGreater(first.visible_cell_count, 0)
        self.assertGreater(first.newly_covered_cell_count, 0)
        self.assertAlmostEqual(first.coverage_rate, first.covered_valid_area / first.total_valid_area)
        self.assertAlmostEqual(first.coverage_rate_delta, first.newly_covered_area / first.total_valid_area)
        self.assertEqual(second.newly_covered_cell_count, 0)
        self.assertEqual(second.coverage_rate_delta, 0.0)
        self.assertEqual(second.covered_valid_cell_count_before, first.covered_valid_cell_count)
        self.assertEqual(second.covered_valid_cell_count, first.covered_valid_cell_count)

    def test_no_valid_cells_return_zero_coverage_without_error(self) -> None:
        grid = generate_sample_grid(width=4, height=3, resolution=1.0)
        grid.layers["valid_mask"][:] = False
        platform = self.sensor_platform(sensor_range=5.0, sensor_fov=90.0)

        report = update_coverage_from_observation(grid, platform, observer_cell=(1, 1), heading_deg=0.0)

        self.assertEqual(report.total_valid_cell_count, 0)
        self.assertEqual(report.total_valid_area, 0.0)
        self.assertEqual(report.coverage_rate, 0.0)
        self.assertEqual(report.coverage_rate_delta, 0.0)
        self.assertEqual(report.newly_covered_cell_count, 0)

    def test_simple_occlusion_excludes_hidden_cells_from_new_coverage(self) -> None:
        grid = generate_sample_grid(width=7, height=5, resolution=1.0)
        grid.layers["obstacle"][:] = 0.0
        grid.layers["obstacle"][2, 3] = 1.0
        platform = self.sensor_platform(sensor_range=6.0, sensor_fov=60.0)

        open_estimate = estimate_observation_coverage(grid, platform, observer_cell=(1, 2), heading_deg=0.0)
        occluded = update_coverage_from_observation(
            grid,
            platform,
            observer_cell=(1, 2),
            heading_deg=0.0,
            use_simple_occlusion=True,
        )

        self.assertIsInstance(open_estimate, CoverageEstimate)
        self.assertIn((5, 2), open_estimate.coverage_cells)
        self.assertNotIn((5, 2), tuple(zip(*np.where(grid.layers["coverage_mask"])[::-1])))
        self.assertLess(occluded.visible_cell_count, open_estimate.visible_cell_count)


if __name__ == "__main__":
    unittest.main()
