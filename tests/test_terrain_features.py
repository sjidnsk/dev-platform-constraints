import unittest

import numpy as np

from dev_platform_constraints.terrain import derive_roughness, derive_slope


class TerrainFeatureTests(unittest.TestCase):
    def test_flat_plane_has_zero_slope_and_zero_roughness(self) -> None:
        elevation = np.full((5, 5), 2.0)
        valid = np.ones_like(elevation, dtype=bool)

        slope, slope_valid = derive_slope(elevation, resolution=1.0, valid_mask=valid)
        roughness, roughness_valid = derive_roughness(elevation, window_size=3, normalization_height=1.0, valid_mask=valid)

        self.assertTrue(np.all(slope_valid))
        self.assertTrue(np.all(roughness_valid))
        self.assertTrue(np.allclose(slope, 0.0))
        self.assertTrue(np.allclose(roughness, 0.0))

    def test_ramp_slope_uses_degrees_and_resolution(self) -> None:
        elevation = np.tile(np.arange(5, dtype=float), (5, 1)) * 0.5
        valid = np.ones_like(elevation, dtype=bool)

        slope, slope_valid = derive_slope(elevation, resolution=1.0, valid_mask=valid)

        self.assertTrue(np.all(slope_valid))
        self.assertAlmostEqual(float(slope[2, 2]), np.degrees(np.arctan(0.5)), places=6)

    def test_step_discontinuity_increases_local_roughness(self) -> None:
        elevation = np.zeros((5, 5), dtype=float)
        elevation[:, 3:] = 2.0
        valid = np.ones_like(elevation, dtype=bool)

        roughness, roughness_valid = derive_roughness(elevation, window_size=3, normalization_height=2.0, valid_mask=valid)

        self.assertTrue(np.all(roughness_valid))
        self.assertGreater(float(roughness[2, 2]), float(roughness[2, 0]))
        self.assertLessEqual(float(np.nanmax(roughness)), 1.0)

    def test_missing_elevation_marks_derived_cells_invalid(self) -> None:
        elevation = np.zeros((4, 4), dtype=float)
        elevation[1, 1] = np.nan
        valid = np.isfinite(elevation)

        slope, slope_valid = derive_slope(elevation, resolution=1.0, valid_mask=valid)
        roughness, roughness_valid = derive_roughness(elevation, window_size=3, normalization_height=1.0, valid_mask=valid)

        self.assertFalse(bool(slope_valid[1, 1]))
        self.assertFalse(bool(roughness_valid[1, 1]))
        self.assertTrue(np.isnan(slope[1, 1]))
        self.assertTrue(np.isnan(roughness[1, 1]))


if __name__ == "__main__":
    unittest.main()
