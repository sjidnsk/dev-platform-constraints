import json
import unittest
from pathlib import Path

from dev_platform_constraints.platforms import (
    VALID_SOURCE_KINDS,
    ParameterValue,
    PlatformParameters,
    default_platform_config_path,
    load_platform_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_CONFIGS = ("yutu.json", "yutu2.json")
REQUIRED_PARAMETERS = (
    "max_slope_deg",
    "max_obstacle_height",
    "ground_clearance",
    "min_turning_radius",
    "sensor_range",
    "sensor_fov",
    "energy_model",
)
REQUIRED_PARAMETER_KEYS = {"value", "unit", "source_kind"}
POSITIVE_FIELDS = ("max_slope_deg", "max_obstacle_height", "ground_clearance", "sensor_range", "sensor_fov")


class PlatformConfigTests(unittest.TestCase):
    def test_yutu_configs_have_required_fields_source_kinds_and_positive_limits(self) -> None:
        for filename in PLATFORM_CONFIGS:
            with self.subTest(config=filename):
                path = REPO_ROOT / "configs" / "platforms" / filename
                raw = json.loads(path.read_text(encoding="utf-8"))

                self.assertIsInstance(raw.get("name"), str)
                self.assertTrue(raw["name"])
                self.assertIsInstance(raw.get("parameters"), dict)

                parameters = raw["parameters"]
                for required in REQUIRED_PARAMETERS:
                    self.assertIn(required, parameters)

                for key, parameter in parameters.items():
                    missing = REQUIRED_PARAMETER_KEYS - parameter.keys()
                    self.assertFalse(missing, f"{filename}:{key} 缺少字段: {sorted(missing)}")
                    self.assertIn(parameter["source_kind"], VALID_SOURCE_KINDS)

                platform = load_platform_parameters(path)
                for field in POSITIVE_FIELDS:
                    self.assertGreater(platform.float_value(field), 0.0, f"{filename}:{field} 必须为正数")

    def test_yutu_configs_warn_for_assumed_parameters(self) -> None:
        for filename in PLATFORM_CONFIGS:
            with self.subTest(config=filename):
                path = REPO_ROOT / "configs" / "platforms" / filename
                platform = load_platform_parameters(path)
                warnings = platform.validate()
                assumed_parameters = [
                    key for key, parameter in platform.parameters.items() if parameter.source_kind == "assumed"
                ]

                self.assertTrue(assumed_parameters)
                for key in assumed_parameters:
                    self.assertIn(
                        f"{key} is assumed and should not be treated as a public fact",
                        warnings,
                    )

    def test_sensor_fov_uses_the_same_positive_constraint_as_other_sensor_limits(self) -> None:
        platform = PlatformParameters(
            name="invalid-sensor-fov",
            parameters={
                "max_slope_deg": ParameterValue(20.0, "deg", "assumed", "测试值"),
                "max_obstacle_height": ParameterValue(0.2, "m", "assumed", "测试值"),
                "ground_clearance": ParameterValue(0.18, "m", "assumed", "测试值"),
                "min_turning_radius": ParameterValue(0.0, "m", "assumed", "P0 占位值"),
                "sensor_range": ParameterValue(5.0, "m", "assumed", "测试值"),
                "sensor_fov": ParameterValue(0.0, "deg", "assumed", "无效测试值"),
                "energy_model": ParameterValue({"base_cost": 1.0, "slope_cost": 1.0}, "relative", "assumed", "测试值"),
            },
        )

        self.assertIn("sensor_fov should be positive", platform.validate())

    def test_default_platform_config_path_points_to_repo_configs(self) -> None:
        self.assertEqual(default_platform_config_path("yutu2"), REPO_ROOT / "configs" / "platforms" / "yutu2.json")
        self.assertTrue(default_platform_config_path("yutu2").exists())


if __name__ == "__main__":
    unittest.main()
