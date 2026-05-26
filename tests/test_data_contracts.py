import unittest

import numpy as np

from dev_platform_constraints.core import CORE_LAYERS, GridMap, LayerMetadata, validate_grid_map


def meta(name: str, unit: str = "unitless") -> LayerMetadata:
    return LayerMetadata(
        source_id=f"test:{name}",
        timestamp=0.0,
        resolution=1.0,
        frame_id="moon_local",
        unit=unit,
    )


class GridMapContractTests(unittest.TestCase):
    def test_core_layers_have_expected_names(self) -> None:
        self.assertEqual(
            CORE_LAYERS,
            (
                "elevation",
                "slope",
                "roughness",
                "obstacle",
                "obstacle_height",
                "illumination",
                "confidence",
                "value",
                "traversability",
                "cost",
                "valid_mask",
            ),
        )

    def test_grid_map_validates_shape_metadata_ranges_and_nan_policy(self) -> None:
        grid = GridMap(resolution=1.0, origin=(0.0, 0.0), width=3, height=2, frame_id="moon_local")
        valid = np.array([[True, False, True], [True, True, True]])
        grid.add_layer("valid_mask", valid, meta("valid_mask"))
        grid.add_layer("elevation", np.array([[0.0, np.nan, 0.2], [0.1, 0.2, 0.3]]), meta("elevation", "m"))
        grid.add_layer("slope", np.zeros((2, 3)), meta("slope", "deg"))
        grid.add_layer("roughness", np.zeros((2, 3)), meta("roughness"))
        grid.add_layer("obstacle", np.zeros((2, 3)), meta("obstacle"))
        grid.add_layer("obstacle_height", np.zeros((2, 3)), meta("obstacle_height", "m"))
        grid.add_layer("illumination", np.ones((2, 3)), meta("illumination"))
        grid.add_layer("confidence", np.full((2, 3), 0.8), meta("confidence"))
        grid.add_layer("value", np.zeros((2, 3)), meta("value"))
        grid.add_layer("traversability", np.ones((2, 3)), meta("traversability"))
        grid.add_layer("cost", np.ones((2, 3)), meta("cost"))

        report = validate_grid_map(grid)

        self.assertTrue(report.is_valid, report.format())
        self.assertEqual(report.missing_layers, ())
        self.assertAlmostEqual(grid.metadata["elevation"].valid_ratio, 5 / 6)

    def test_valid_cells_cannot_contain_nan_and_normalized_layers_are_bounded(self) -> None:
        grid = GridMap(resolution=1.0, origin=(0.0, 0.0), width=2, height=2, frame_id="moon_local")
        valid = np.ones((2, 2), dtype=bool)
        grid.add_layer("valid_mask", valid, meta("valid_mask"))
        for layer in CORE_LAYERS:
            if layer == "valid_mask":
                continue
            data = np.zeros((2, 2), dtype=float)
            grid.add_layer(layer, data, meta(layer))
        grid.layers["confidence"][0, 0] = 1.2
        grid.layers["elevation"][1, 1] = np.nan

        report = validate_grid_map(grid)

        self.assertFalse(report.is_valid)
        self.assertIn("confidence", report.format())
        self.assertIn("NaN", report.format())


if __name__ == "__main__":
    unittest.main()
