from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class ValidationMapSpec:
    scenario_id: str
    width: int
    height: int
    resolution: float
    seed: int
    observations: tuple[dict[str, object], ...]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    low_confidence_band: tuple[int, int]
    value_region: tuple[int, int, int, int]
    risk_region: tuple[int, int, int, int] | None = None
    blocked_rects: tuple[tuple[int, int, int, int], ...] = tuple()
    scenario_group: str = "smoke"

    @property
    def filename(self) -> str:
        return f"{self.scenario_id}.npz"


SMOKE_VALIDATION_SPECS = (
    ValidationMapSpec(
        scenario_id="npz_shadow_corridor",
        width=16,
        height=10,
        resolution=0.5,
        seed=401,
        observations=({"observer_cell": [0, 5], "heading_deg": 0.0},),
        start_cell=(0, 0),
        goal_cell=(15, 9),
        low_confidence_band=(5, 8),
        value_region=(12, 16, 7, 10),
        risk_region=(6, 9, 2, 8),
    ),
    ValidationMapSpec(
        scenario_id="npz_rock_field_multi_pose",
        width=24,
        height=14,
        resolution=0.5,
        seed=402,
        observations=(
            {"observer_cell": [0, 4], "heading_deg": 0.0},
            {"observer_cell": [0, 10], "heading_deg": 0.0},
        ),
        start_cell=(0, 1),
        goal_cell=(23, 13),
        low_confidence_band=(7, 11),
        value_region=(18, 24, 9, 14),
    ),
    ValidationMapSpec(
        scenario_id="npz_low_confidence_risk_band",
        width=18,
        height=16,
        resolution=0.5,
        seed=403,
        observations=({"observer_cell": [0, 8], "heading_deg": 0.0},),
        start_cell=(0, 0),
        goal_cell=(17, 15),
        low_confidence_band=(6, 10),
        value_region=(13, 18, 11, 16),
        risk_region=(6, 10, 0, 16),
    ),
)

STRESS_VALIDATION_SPECS = (
    ValidationMapSpec(
        scenario_id="npz_near_blocked_corridor",
        width=20,
        height=12,
        resolution=0.5,
        seed=501,
        observations=({"observer_cell": [0, 6], "heading_deg": 0.0},),
        start_cell=(1, 6),
        goal_cell=(19, 8),
        low_confidence_band=(7, 11),
        value_region=(15, 20, 4, 10),
        risk_region=(7, 11, 0, 12),
        blocked_rects=((9, 10, 0, 6), (9, 10, 7, 12)),
        scenario_group="stress",
    ),
    ValidationMapSpec(
        scenario_id="npz_high_risk_value_trap",
        width=22,
        height=14,
        resolution=0.5,
        seed=502,
        observations=({"observer_cell": [0, 3], "heading_deg": 4.0},),
        start_cell=(1, 3),
        goal_cell=(21, 12),
        low_confidence_band=(8, 13),
        value_region=(16, 22, 8, 14),
        risk_region=(8, 16, 5, 12),
        blocked_rects=((10, 12, 4, 9), (13, 15, 7, 12)),
        scenario_group="stress",
    ),
    ValidationMapSpec(
        scenario_id="npz_dense_rock_choke",
        width=24,
        height=16,
        resolution=0.5,
        seed=503,
        observations=(
            {"observer_cell": [0, 6], "heading_deg": 0.0},
            {"observer_cell": [0, 11], "heading_deg": 0.0},
        ),
        start_cell=(1, 8),
        goal_cell=(23, 10),
        low_confidence_band=(9, 14),
        value_region=(18, 24, 5, 14),
        risk_region=(9, 15, 4, 13),
        blocked_rects=(
            (8, 9, 0, 7),
            (8, 9, 9, 16),
            (12, 13, 2, 10),
            (12, 13, 12, 16),
            (16, 17, 0, 5),
            (16, 17, 7, 16),
        ),
        scenario_group="stress",
    ),
    ValidationMapSpec(
        scenario_id="npz_path_complexity_benefit_probe",
        width=28,
        height=18,
        resolution=0.5,
        seed=505,
        observations=(
            {"observer_cell": [1, 9], "heading_deg": 0.0},
            {"observer_cell": [6, 2], "heading_deg": 10.0},
        ),
        start_cell=(1, 9),
        goal_cell=(27, 9),
        low_confidence_band=(8, 20),
        value_region=(22, 28, 6, 13),
        risk_region=(8, 20, 4, 14),
        blocked_rects=(
            (11, 12, 4, 10),
            (11, 12, 12, 14),
            (15, 16, 4, 8),
            (15, 16, 10, 14),
            (19, 20, 6, 12),
        ),
        scenario_group="stress",
    ),
    ValidationMapSpec(
        scenario_id="npz_mixed_stress_detour",
        width=26,
        height=14,
        resolution=0.5,
        seed=504,
        observations=(
            {"observer_cell": [1, 6], "heading_deg": 0.0},
            {"observer_cell": [7, 3], "heading_deg": 12.0},
        ),
        start_cell=(1, 6),
        goal_cell=(25, 10),
        low_confidence_band=(3, 14),
        value_region=(3, 25, 3, 12),
        risk_region=(10, 18, 4, 11),
        blocked_rects=(
            (11, 13, 2, 8),
            (11, 13, 10, 14),
            (18, 20, 0, 5),
            (18, 20, 7, 14),
            (22, 24, 4, 10),
        ),
        scenario_group="mixed_stress",
    ),
)

SCENARIO_SETS = {
    "smoke": SMOKE_VALIDATION_SPECS,
    "stress": STRESS_VALIDATION_SPECS,
    "all": SMOKE_VALIDATION_SPECS + STRESS_VALIDATION_SPECS,
}
VALIDATION_SPECS = SMOKE_VALIDATION_SPECS


def _validation_layers(spec: ValidationMapSpec) -> dict[str, np.ndarray | float | tuple[float, float]]:
    rng = np.random.default_rng(spec.seed)
    yy, xx = np.mgrid[0 : spec.height, 0 : spec.width]
    x_norm = xx / max(spec.width - 1, 1)
    y_norm = yy / max(spec.height - 1, 1)

    elevation = 0.018 * xx + 0.012 * yy + 0.035 * np.sin(xx / 3.2) + 0.020 * np.cos(yy / 2.7)
    for center_x, center_y, radius, depth in (
        (0.35, 0.55, 0.030, 0.055),
        (0.72, 0.35, 0.025, 0.040),
    ):
        crater = np.exp(-(((x_norm - center_x) ** 2 + (y_norm - center_y) ** 2) / radius))
        elevation -= depth * crater
    elevation += rng.normal(0.0, 0.002, size=(spec.height, spec.width))

    obstacle = np.zeros((spec.height, spec.width), dtype=float)
    rock_count = max(2, (spec.width * spec.height) // 55)
    rock_xs = rng.integers(max(2, spec.width // 4), max(3, spec.width - 2), size=rock_count)
    rock_ys = rng.integers(1, max(2, spec.height - 1), size=rock_count)
    obstacle[rock_ys, rock_xs] = rng.uniform(0.25, 0.45, size=rock_count)
    obstacle_height = obstacle * 0.18

    shadow_center = rng.uniform(0.35, 0.70)
    shadow_width = rng.uniform(0.035, 0.075)
    shadow_band = np.exp(-((y_norm - shadow_center) ** 2) / shadow_width)
    illumination = np.clip(0.86 - 0.34 * shadow_band + 0.06 * np.cos(x_norm * np.pi), 0.08, 1.0)

    confidence = np.clip(0.72 + rng.normal(0.0, 0.015, size=(spec.height, spec.width)), 0.55, 0.82)
    x0, x1 = spec.low_confidence_band
    confidence[:, x0:x1] = np.minimum(confidence[:, x0:x1], 0.42)
    if spec.risk_region is not None:
        rx0, rx1, ry0, ry1 = spec.risk_region
        illumination[ry0:ry1, rx0:rx1] = np.minimum(illumination[ry0:ry1, rx0:rx1], 0.28)
        confidence[ry0:ry1, rx0:rx1] = np.minimum(confidence[ry0:ry1, rx0:rx1], 0.32)
    for x0, x1, y0, y1 in spec.blocked_rects:
        obstacle[y0:y1, x0:x1] = 1.0
        obstacle_height[y0:y1, x0:x1] = 0.28
        illumination[y0:y1, x0:x1] = np.minimum(illumination[y0:y1, x0:x1], 0.20)
        confidence[y0:y1, x0:x1] = np.minimum(confidence[y0:y1, x0:x1], 0.25)

    value = np.zeros((spec.height, spec.width), dtype=float)
    vx0, vx1, vy0, vy1 = spec.value_region
    value[vy0:vy1, vx0:vx1] = 0.8
    valid_mask = np.ones((spec.height, spec.width), dtype=bool)

    return {
        "resolution": spec.resolution,
        "origin": (0.0, 0.0),
        "elevation": elevation,
        "obstacle": obstacle,
        "obstacle_height": obstacle_height,
        "illumination": illumination,
        "confidence": confidence,
        "value": value,
        "valid_mask": valid_mask,
    }


def _scenario_entry(spec: ValidationMapSpec, map_path: Path) -> dict[str, object]:
    scenario: dict[str, object] = {
        "scenario_id": spec.scenario_id,
        "width": spec.width,
        "height": spec.height,
        "resolution": spec.resolution,
        "map_source": {"kind": "npz_grid", "path": str(map_path)},
        "observations": list(spec.observations),
        "start_cell": list(spec.start_cell),
        "goal_cell": list(spec.goal_cell),
        "elapsed_time": 1.5,
        "recency_time_constant": 10.0,
        "low_confidence_band": list(spec.low_confidence_band),
        "value_region": list(spec.value_region),
        "scenario_group": spec.scenario_group,
    }
    if spec.risk_region is not None:
        scenario["risk_region"] = list(spec.risk_region)
    if spec.blocked_rects:
        scenario["blocked_rects"] = [list(rect) for rect in spec.blocked_rects]
    return scenario


def _specs_for_set(scenario_set: str) -> tuple[ValidationMapSpec, ...]:
    try:
        return SCENARIO_SETS[scenario_set]
    except KeyError as exc:
        raise ValueError(f"unknown scenario set: {scenario_set}") from exc


def build_scenario_config(output_dir: Path, scenario_set: str = "smoke") -> dict[str, object]:
    specs = _specs_for_set(scenario_set)
    return {
        "terrain_likelihood_config": str(ROOT / "configs" / "confidence" / "terrain_likelihood_default.json"),
        "scenario_set": scenario_set,
        "scenarios": [_scenario_entry(spec, output_dir / spec.filename) for spec in specs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成可复现的 npz_grid 外部地图验证集。")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "validation_maps"), help="输出 .npz 地图目录。")
    parser.add_argument("--scenario-config", default=None, help="可选：同时写出场景配置 JSON。")
    parser.add_argument(
        "--scenario-set",
        choices=tuple(SCENARIO_SETS),
        default="smoke",
        help="选择要生成的验证场景集：smoke、stress 或 all。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将生成的地图和场景，不写文件。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    specs = _specs_for_set(args.scenario_set)
    scenario_config = build_scenario_config(output_dir, args.scenario_set)
    summary = {
        "output_dir": str(output_dir),
        "scenario_config": str(args.scenario_config) if args.scenario_config else None,
        "scenario_set": args.scenario_set,
        "scenarios": [
            {
                "scenario_id": spec.scenario_id,
                "path": str(output_dir / spec.filename),
                "width": spec.width,
                "height": spec.height,
                "seed": spec.seed,
                "scenario_group": spec.scenario_group,
            }
            for spec in specs
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        np.savez(output_dir / spec.filename, **_validation_layers(spec))

    if args.scenario_config:
        scenario_path = Path(args.scenario_config)
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_text(json.dumps(scenario_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
