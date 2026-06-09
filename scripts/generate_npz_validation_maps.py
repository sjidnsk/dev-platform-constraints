from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
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
    contrast_focus: str | None = None

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
    ValidationMapSpec(
        scenario_id="npz_low_centerline_bad_channel",
        width=20,
        height=16,
        resolution=0.5,
        seed=506,
        observations=({"observer_cell": [0, 8], "heading_deg": 0.0},),
        start_cell=(0, 0),
        goal_cell=(19, 15),
        low_confidence_band=(6, 11),
        value_region=(14, 20, 10, 16),
        risk_region=(6, 11, 0, 16),
        scenario_group="channel_contrast",
        contrast_focus="low_centerline_cost_bad_channel_quality",
    ),
    ValidationMapSpec(
        scenario_id="npz_blocked_nearby_clearance_detour",
        width=24,
        height=16,
        resolution=0.5,
        seed=507,
        observations=(
            {"observer_cell": [0, 5], "heading_deg": 0.0},
            {"observer_cell": [0, 12], "heading_deg": 0.0},
        ),
        start_cell=(0, 2),
        goal_cell=(23, 14),
        low_confidence_band=(7, 16),
        value_region=(17, 24, 10, 16),
        risk_region=(7, 15, 4, 12),
        blocked_rects=(
            (4, 8, 3, 8),
        ),
        scenario_group="channel_contrast",
        contrast_focus="blocked_nearby_clearance",
    ),
    ValidationMapSpec(
        scenario_id="npz_high_cost_exposure_rock_detour",
        width=24,
        height=16,
        resolution=0.5,
        seed=508,
        observations=(
            {"observer_cell": [0, 5], "heading_deg": 0.0},
            {"observer_cell": [0, 12], "heading_deg": 0.0},
        ),
        start_cell=(0, 1),
        goal_cell=(23, 14),
        low_confidence_band=(6, 15),
        value_region=(17, 24, 9, 16),
        risk_region=(6, 15, 3, 12),
        blocked_rects=(
            (4, 8, 3, 9),
        ),
        scenario_group="channel_contrast",
        contrast_focus="high_cost_exposure_rock_field_detour",
    ),
)

HOLDOUT_VALIDATION_SPECS = (
    ValidationMapSpec(
        scenario_id="npz_holdout_near_blocked_corridor",
        width=21,
        height=13,
        resolution=0.5,
        seed=8601,
        observations=({"observer_cell": [0, 6], "heading_deg": 2.0},),
        start_cell=(1, 6),
        goal_cell=(20, 9),
        low_confidence_band=(8, 12),
        value_region=(16, 21, 5, 11),
        risk_region=(8, 12, 1, 12),
        blocked_rects=((10, 11, 0, 6), (10, 11, 8, 13)),
        scenario_group="holdout_near_blocked",
    ),
    ValidationMapSpec(
        scenario_id="npz_holdout_high_risk_value_trap",
        width=23,
        height=15,
        resolution=0.5,
        seed=8602,
        observations=({"observer_cell": [0, 4], "heading_deg": 6.0},),
        start_cell=(1, 4),
        goal_cell=(22, 13),
        low_confidence_band=(9, 14),
        value_region=(17, 23, 9, 15),
        risk_region=(9, 17, 5, 13),
        blocked_rects=((11, 13, 4, 10), (14, 16, 8, 13)),
        scenario_group="holdout_high_risk",
    ),
    ValidationMapSpec(
        scenario_id="npz_holdout_dense_rock_choke",
        width=25,
        height=17,
        resolution=0.5,
        seed=8603,
        observations=(
            {"observer_cell": [0, 6], "heading_deg": 0.0},
            {"observer_cell": [0, 12], "heading_deg": 0.0},
        ),
        start_cell=(1, 9),
        goal_cell=(24, 11),
        low_confidence_band=(10, 15),
        value_region=(19, 25, 6, 15),
        risk_region=(10, 16, 4, 14),
        blocked_rects=(
            (8, 9, 0, 8),
            (8, 9, 10, 17),
            (13, 14, 2, 11),
            (13, 14, 13, 17),
            (17, 18, 0, 6),
            (17, 18, 8, 17),
        ),
        scenario_group="holdout_dense_choke",
    ),
    ValidationMapSpec(
        scenario_id="npz_holdout_path_complexity_probe",
        width=29,
        height=19,
        resolution=0.5,
        seed=8604,
        observations=(
            {"observer_cell": [1, 10], "heading_deg": 0.0},
            {"observer_cell": [7, 3], "heading_deg": 8.0},
        ),
        start_cell=(1, 10),
        goal_cell=(28, 10),
        low_confidence_band=(9, 21),
        value_region=(23, 29, 7, 14),
        risk_region=(9, 21, 5, 15),
        blocked_rects=(
            (12, 13, 5, 11),
            (12, 13, 13, 15),
            (16, 17, 5, 9),
            (16, 17, 11, 15),
            (20, 21, 7, 13),
        ),
        scenario_group="holdout_path_complexity",
    ),
    ValidationMapSpec(
        scenario_id="npz_holdout_channel_contrast_detour",
        width=25,
        height=17,
        resolution=0.5,
        seed=8605,
        observations=(
            {"observer_cell": [0, 5], "heading_deg": 0.0},
            {"observer_cell": [0, 13], "heading_deg": 0.0},
        ),
        start_cell=(0, 2),
        goal_cell=(24, 15),
        low_confidence_band=(7, 16),
        value_region=(18, 25, 11, 17),
        risk_region=(7, 16, 4, 13),
        blocked_rects=((5, 9, 3, 9),),
        scenario_group="holdout_channel_contrast",
        contrast_focus="holdout_blocked_nearby_clearance",
    ),
)


def _raw_alignment_specs(split: str, seed_offset: int) -> tuple[ValidationMapSpec, ...]:
    specs: list[ValidationMapSpec] = []
    for spec in HOLDOUT_VALIDATION_SPECS:
        base_id = spec.scenario_id.removeprefix("npz_holdout_")
        specs.append(
            replace(
                spec,
                scenario_id=f"npz_raw_align_{split}_{base_id}",
                seed=spec.seed + seed_offset,
                scenario_group=f"raw_align_{split}_{base_id}",
                contrast_focus=(
                    f"raw_align_{split}_{spec.contrast_focus}"
                    if spec.contrast_focus is not None
                    else None
                ),
            )
        )
    return tuple(specs)


RAW_ALIGN_TRAIN_VALIDATION_SPECS = _raw_alignment_specs("train", 1100)
RAW_ALIGN_VAL_VALIDATION_SPECS = _raw_alignment_specs("val", 2100)
RAW_ALIGN_TEST_VALIDATION_SPECS = _raw_alignment_specs("test", 3100)


POLICY_CANARY_VALIDATION_SPECS = (
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
        contrast_focus="safe_alternative_policy_choice",
    ),
)

POLICY_CANARY_DIVERSITY_VALIDATION_SPECS = (
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id="npz_canary_mixed_stress_detour_v2",
        seed=9504,
        scenario_group="mixed_stress_detour",
        contrast_focus="safe_alternative_policy_choice",
    ),
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id="npz_canary_near_blocked_safe_alt",
        seed=9505,
        scenario_group="near_blocked_safe_alt",
        contrast_focus="near_blocked_safe_alternative",
    ),
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id="npz_canary_high_risk_tradeoff",
        seed=9506,
        scenario_group="high_risk_tradeoff",
        contrast_focus="high_risk_safe_tradeoff",
    ),
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id="npz_canary_dense_choke_safe_bypass",
        seed=9507,
        scenario_group="dense_choke_safe_bypass",
        contrast_focus="dense_choke_safe_bypass",
    ),
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id="npz_canary_channel_contrast",
        seed=9508,
        scenario_group="channel_contrast",
        contrast_focus="channel_quality_safe_alternative",
    ),
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id="npz_canary_path_complexity_benefit",
        seed=9509,
        scenario_group="path_complexity_benefit",
        contrast_focus="path_complexity_safe_benefit",
    ),
)

_POLICY_CANARY_OPPORTUNITY_QUALITY_FAMILIES = (
    ("mixed_stress_detour", "safe_alternative_policy_choice", (9504, 9605)),
    ("near_blocked_safe_alt", "near_blocked_safe_alternative", (9606, 9607)),
    ("channel_contrast", "channel_quality_safe_alternative", (9608, 9609)),
    ("high_risk_tradeoff", "high_risk_safe_tradeoff", (9610, 9611)),
    ("dense_choke_safe_bypass", "dense_choke_safe_bypass", (9612, 9613)),
    ("path_complexity_benefit", "path_complexity_safe_benefit", (9614, 9615)),
)

POLICY_CANARY_OPPORTUNITY_QUALITY_VALIDATION_SPECS = tuple(
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id=f"npz_canary_opportunity_quality_{family}_{variant}",
        seed=seeds[variant_index],
        scenario_group=family,
        contrast_focus=contrast_focus,
    )
    for family, contrast_focus, seeds in (
        _POLICY_CANARY_OPPORTUNITY_QUALITY_FAMILIES
    )
    for variant_index, variant in enumerate(("a", "b"))
)

_DENSE_CHOKE_SAFE_BYPASS_BLOCKS = (
    (11, 13, 2, 8),
    (11, 13, 10, 14),
    (18, 20, 0, 5),
    (18, 20, 7, 14),
    (22, 24, 4, 10),
)

POLICY_CANARY_DENSE_CHOKE_OPPORTUNITY_VALIDATION_SPECS = tuple(
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id=f"npz_canary_dense_choke_safe_bypass_{variant}",
        seed=9704 + index,
        scenario_group="dense_choke_safe_bypass",
        contrast_focus="dense_choke_safe_bypass",
        blocked_rects=_DENSE_CHOKE_SAFE_BYPASS_BLOCKS,
        risk_region=(10, 18, 4, 11),
        value_region=(4, 25, 3, 12),
        low_confidence_band=(3, 14),
    )
    for index, variant in enumerate(("a", "b", "c", "d"))
)


_POLICY_CANARY_VALUE_STABILITY_FAMILIES = (
    ("mixed_stress_detour", "safe_alternative_policy_choice", 9800, (0, 0)),
    ("near_blocked_safe_alt", "near_blocked_safe_alternative", 9810, (1, 0)),
    ("high_risk_tradeoff", "high_risk_safe_tradeoff", 9820, (0, 1)),
    ("dense_choke_safe_bypass", "dense_choke_safe_bypass", 9830, (-1, 0)),
    ("channel_contrast", "channel_quality_safe_alternative", 9840, (1, 1)),
    ("path_complexity_benefit", "path_complexity_safe_benefit", 9850, (-1, 1)),
)

_POLICY_CANARY_VALUE_STABILITY_VARIANTS = (
    {
        "suffix": "a",
        "low_confidence_band": (3, 14),
        "value_region": (3, 25, 3, 12),
        "risk_region": (10, 18, 4, 11),
        "blocked_rects": (
            (11, 13, 2, 8),
            (11, 13, 10, 14),
            (18, 20, 0, 5),
            (18, 20, 7, 14),
            (22, 24, 4, 10),
        ),
    },
    {
        "suffix": "b",
        "low_confidence_band": (4, 15),
        "value_region": (4, 25, 3, 12),
        "risk_region": (10, 17, 4, 11),
        "blocked_rects": (
            (10, 12, 2, 7),
            (10, 12, 9, 14),
            (17, 19, 0, 5),
            (17, 19, 7, 14),
            (21, 23, 4, 10),
        ),
    },
    {
        "suffix": "c",
        "low_confidence_band": (3, 13),
        "value_region": (3, 24, 2, 12),
        "risk_region": (9, 18, 3, 10),
        "blocked_rects": (
            (11, 13, 1, 7),
            (11, 13, 9, 14),
            (18, 20, 0, 4),
            (18, 20, 6, 14),
            (22, 24, 3, 9),
        ),
    },
    {
        "suffix": "d",
        "low_confidence_band": (5, 15),
        "value_region": (5, 25, 4, 13),
        "risk_region": (11, 19, 4, 12),
        "blocked_rects": (
            (12, 14, 2, 8),
            (12, 14, 10, 14),
            (19, 21, 1, 6),
            (19, 21, 8, 14),
            (22, 24, 5, 11),
        ),
    },
    {
        "suffix": "e",
        "low_confidence_band": (3, 15),
        "value_region": (3, 25, 2, 11),
        "risk_region": (10, 19, 3, 11),
        "blocked_rects": (
            (10, 13, 2, 8),
            (10, 13, 10, 14),
            (18, 21, 0, 5),
            (18, 21, 7, 14),
            (22, 25, 4, 10),
        ),
    },
    {
        "suffix": "f",
        "low_confidence_band": (4, 14),
        "value_region": (4, 25, 4, 13),
        "risk_region": (9, 17, 4, 12),
        "blocked_rects": (
            (11, 12, 2, 8),
            (11, 12, 10, 14),
            (17, 20, 0, 5),
            (17, 20, 7, 14),
            (21, 24, 4, 10),
        ),
    },
)


def _shift_interval(interval: tuple[int, int], dx: int, *, lower: int, upper: int) -> tuple[int, int]:
    width = interval[1] - interval[0]
    start = min(max(interval[0] + dx, lower), upper - width)
    return (start, start + width)


def _shift_region(
    region: tuple[int, int, int, int],
    dx: int,
    dy: int,
) -> tuple[int, int, int, int]:
    x0, x1 = _shift_interval((region[0], region[1]), dx, lower=0, upper=26)
    y0, y1 = _shift_interval((region[2], region[3]), dy, lower=0, upper=14)
    return (x0, x1, y0, y1)


def _shift_rects(
    rects: tuple[tuple[int, int, int, int], ...],
    dx: int,
    dy: int,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(_shift_region(rect, dx, dy) for rect in rects)


POLICY_CANARY_VALUE_STABILITY_VALIDATION_SPECS = tuple(
    replace(
        POLICY_CANARY_VALIDATION_SPECS[0],
        scenario_id=(
            f"npz_canary_value_stability_{family}_{variant['suffix']}"
        ),
        seed=seed_base + variant_index,
        scenario_group=family,
        contrast_focus=contrast_focus,
        low_confidence_band=_shift_interval(
            variant["low_confidence_band"],
            offset[0],
            lower=0,
            upper=26,
        ),
        value_region=_shift_region(variant["value_region"], offset[0], offset[1]),
        risk_region=_shift_region(variant["risk_region"], offset[0], offset[1]),
        blocked_rects=_shift_rects(variant["blocked_rects"], offset[0], offset[1]),
    )
    for family, contrast_focus, seed_base, offset in _POLICY_CANARY_VALUE_STABILITY_FAMILIES
    for variant_index, variant in enumerate(_POLICY_CANARY_VALUE_STABILITY_VARIANTS)
)


SCENARIO_SETS = {
    "smoke": SMOKE_VALIDATION_SPECS,
    "stress": STRESS_VALIDATION_SPECS,
    "holdout": HOLDOUT_VALIDATION_SPECS,
    "raw_align_train": RAW_ALIGN_TRAIN_VALIDATION_SPECS,
    "raw_align_val": RAW_ALIGN_VAL_VALIDATION_SPECS,
    "raw_align_test": RAW_ALIGN_TEST_VALIDATION_SPECS,
    "policy_canary": POLICY_CANARY_VALIDATION_SPECS,
    "policy_canary_diversity": POLICY_CANARY_DIVERSITY_VALIDATION_SPECS,
    "policy_canary_opportunity_quality": POLICY_CANARY_OPPORTUNITY_QUALITY_VALIDATION_SPECS,
    "policy_canary_dense_choke_opportunity": POLICY_CANARY_DENSE_CHOKE_OPPORTUNITY_VALIDATION_SPECS,
    "policy_canary_value_stability": POLICY_CANARY_VALUE_STABILITY_VALIDATION_SPECS,
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
        "seed": spec.seed,
        "scenario_variant_id": f"{spec.scenario_id}-seed-{spec.seed}",
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
    if spec.contrast_focus is not None:
        scenario["contrast_focus"] = spec.contrast_focus
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
        help=(
            "选择要生成的验证场景集：smoke、stress、holdout、raw_align_train、"
            "raw_align_val、raw_align_test、policy_canary、policy_canary_diversity、"
            "policy_canary_opportunity_quality、policy_canary_dense_choke_opportunity、"
            "policy_canary_value_stability 或 all。"
        ),
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
                "scenario_variant_id": f"{spec.scenario_id}-seed-{spec.seed}",
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
