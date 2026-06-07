from __future__ import annotations

from dataclasses import asdict, is_dataclass
from math import ceil, hypot
from typing import Any

import numpy as np

from ..core import ValidationReport
from ..core.layers import GridMap
from ..exploration import GoalSequenceEvaluation, ScoredGoal
from ..mapping.constraints import ConstraintResult


MODEL_EXPLORER_SCHEMA_VERSION = "model-explorer-contract/v1"
MODEL_EXPLORER_STABLE_FIELDS = (
    "schema_version",
    "grid.width",
    "grid.height",
    "grid.resolution",
    "grid.frame_id",
    "grid.origin",
    "grid.layers",
    "constraints.violation_count",
    "constraints.passable_ratio",
    "constraints.reason_counts",
    "top_goals.cell",
    "top_goals.utility",
    "top_goals.reachable",
    "top_sequences.cells",
    "top_sequences.utility",
    "top_sequences.coverage_area",
    "observation_update",
)


def build_data_contract_report(grid: GridMap, validation_report: ValidationReport) -> dict[str, Any]:
    """构建可序列化的数据契约检查报告。"""

    layers: dict[str, dict[str, object]] = {}
    for name in sorted(grid.layers):
        metadata = grid.metadata.get(name)
        layers[name] = {
            "shape": list(grid.layers[name].shape),
            "source_id": metadata.source_id if metadata is not None else None,
            "timestamp": metadata.timestamp if metadata is not None else None,
            "resolution": metadata.resolution if metadata is not None else None,
            "frame_id": metadata.frame_id if metadata is not None else None,
            "unit": metadata.unit if metadata is not None else None,
            "valid_ratio": metadata.valid_ratio if metadata is not None else None,
            "source_kind": metadata.source_kind if metadata is not None else None,
        }

    errors = [issue.format() for issue in validation_report.errors]
    warnings = [issue.format() for issue in validation_report.warnings]
    return {
        "is_valid": validation_report.is_valid,
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "resolution": grid.resolution,
            "frame_id": grid.frame_id,
            "origin": list(grid.origin),
        },
        "layers": layers,
        "missing_layers": list(validation_report.missing_layers),
        "issues": {
            "errors": errors,
            "warnings": warnings,
        },
        "issue_summary": {
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }


def _reason_counts(constraints: ConstraintResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code, name in sorted(constraints.reason_names.items()):
        counts[name] = int(np.count_nonzero(np.bitwise_and(constraints.reason_layer, code)))
    return counts


def _observation_update_payload(confidence_report: Any) -> dict[str, object]:
    if confidence_report is None:
        return {}
    if isinstance(confidence_report, dict):
        return dict(confidence_report)
    if is_dataclass(confidence_report):
        return asdict(confidence_report)
    return {
        name: value
        for name in dir(confidence_report)
        if not name.startswith("_") and not callable(value := getattr(confidence_report, name))
    }


def _platform_parameter_float(platform_parameters: Any | None, key: str) -> float | None:
    if platform_parameters is None:
        return None
    parameters = getattr(platform_parameters, "parameters", {})
    parameter = parameters.get(key) if isinstance(parameters, dict) else None
    if parameter is None:
        return None
    value = getattr(parameter, "value", parameter)
    if isinstance(value, dict):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _platform_footprint_radius_m(
    platform_parameters: Any | None,
    *,
    safety_margin_m: float = 0.0,
) -> float | None:
    body_length_m = _platform_parameter_float(platform_parameters, "body_length")
    body_width_m = _platform_parameter_float(platform_parameters, "body_width")
    if body_length_m is None or body_width_m is None:
        return None
    return float(hypot(body_length_m, body_width_m) / 2.0 + safety_margin_m)


def _inflated_passable_mask(
    passable_mask: np.ndarray,
    *,
    resolution: float,
    footprint_radius_m: float | None,
) -> np.ndarray:
    safe_mask = np.array(passable_mask, dtype=bool, copy=True)
    if footprint_radius_m is None or footprint_radius_m <= 0.0:
        return safe_mask

    radius_cells = int(ceil(footprint_radius_m / max(resolution, 1.0e-12)))
    height, width = safe_mask.shape
    for blocked_y, blocked_x in np.argwhere(~passable_mask):
        min_y = max(0, int(blocked_y) - radius_cells)
        max_y = min(height - 1, int(blocked_y) + radius_cells)
        min_x = max(0, int(blocked_x) - radius_cells)
        max_x = min(width - 1, int(blocked_x) + radius_cells)
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance_m = hypot((x - int(blocked_x)) * resolution, (y - int(blocked_y)) * resolution)
                if distance_m <= footprint_radius_m:
                    safe_mask[y, x] = False
    return safe_mask


def _platform_goal_admissibility_payload(
    passable_mask: np.ndarray,
    *,
    resolution: float,
    platform_parameters: Any | None,
    safety_margin_m: float,
) -> dict[str, Any]:
    if safety_margin_m < 0.0:
        raise ValueError("safety_margin_m must be nonnegative")

    footprint_radius_m = _platform_footprint_radius_m(
        platform_parameters,
        safety_margin_m=safety_margin_m,
    )
    inflated_mask = _inflated_passable_mask(
        passable_mask,
        resolution=resolution,
        footprint_radius_m=footprint_radius_m,
    )
    uses_platform_footprint = footprint_radius_m is not None and footprint_radius_m > 0.0
    return {
        "schema_version": "platform-goal-admissibility/v1",
        "passable_source": "inflated_passable_mask" if uses_platform_footprint else "original_passable_mask",
        "resolution": float(resolution),
        "safety_margin_m": float(safety_margin_m),
        "footprint_radius_m": footprint_radius_m,
        "original_blocked_count": int(np.count_nonzero(~passable_mask)),
        "inflated_blocked_count": int(np.count_nonzero(~inflated_mask)),
        "inflated_passable_mask": inflated_mask.tolist(),
        "cell_roles": {
            "policy_target_cell": "model_explorer_contract_top_goal",
            "execution_goal_cell": "same_cell_when_inflated_passable",
            "nearest_inflated_passable_anchor": (
                "audit_projection_candidate_when_policy_target_is_not_inflated_passable"
            ),
        },
        "training_use": {
            "same_cell_inflated_passable_goal": "eligible_if_route_contract_passes",
            "platform_inflated_goal_blocked": "not_positive_evidence",
            "audit_proxy_anchor_not_same_cell": "not_positive_evidence",
        },
    }


def build_model_explorer_contract(
    grid: GridMap,
    constraints: ConstraintResult,
    scored_goals: tuple[ScoredGoal, ...],
    goal_sequences: tuple[GoalSequenceEvaluation, ...],
    confidence_report: Any,
) -> dict[str, Any]:
    """构建供 model-explorer 消费的稳定 JSON 契约摘要。"""

    passable_mask = np.asarray(constraints.passable_mask, dtype=bool)
    return {
        "schema_version": MODEL_EXPLORER_SCHEMA_VERSION,
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "resolution": grid.resolution,
            "frame_id": grid.frame_id,
            "origin": list(grid.origin),
            "layers": sorted(grid.layers),
        },
        "constraints": {
            "violation_count": constraints.violation_count,
            "passable_ratio": float(np.mean(passable_mask)) if passable_mask.size else 0.0,
            "reason_counts": _reason_counts(constraints),
        },
        "top_goals": [
            {
                "cell": list(goal.candidate.cell),
                "utility": float(goal.utility),
                "reachable": bool(goal.candidate.reachable),
                "information_gain": float(goal.candidate.information_gain),
                "value": float(goal.candidate.value),
                "confidence_gain": float(goal.candidate.confidence_gain),
                "risk": float(goal.candidate.risk),
                "path_cost": float(goal.candidate.path_cost),
                "energy_cost": float(goal.candidate.energy_cost),
                "coverage_area": float(goal.candidate.coverage_area),
                "expected_new_coverage_area": float(goal.candidate.expected_new_coverage_area),
                "expected_coverage_rate_delta": float(goal.candidate.expected_coverage_rate_delta),
            }
            for goal in scored_goals
        ],
        "top_sequences": [
            {
                "cells": [list(goal.cell) for goal in sequence.goals],
                "utility": float(sequence.utility),
                "delta_c": float(sequence.delta_c),
                "value_coverage": float(sequence.value_coverage),
                "risk": float(sequence.risk),
                "path_cost": float(sequence.path_cost),
                "reachable": bool(sequence.reachable),
                "coverage_area": float(sequence.coverage_area),
                "segment_path_costs": [float(value) for value in sequence.segment_path_costs],
                "cumulative_risk": float(sequence.cumulative_risk),
                "unreachable_reasons": list(sequence.unreachable_reasons),
                "risk_reasons": list(sequence.risk_reasons),
            }
            for sequence in goal_sequences
        ],
        "observation_update": _observation_update_payload(confidence_report),
        "stable_fields": list(MODEL_EXPLORER_STABLE_FIELDS),
        "experimental_fields": [
            "top_goals.information_gain",
            "top_goals.value",
            "top_goals.confidence_gain",
            "top_goals.risk",
            "top_goals.path_cost",
            "top_goals.energy_cost",
            "top_goals.coverage_area",
            "top_goals.expected_new_coverage_area",
            "top_goals.expected_coverage_rate_delta",
            "observation_update.total_valid_area",
            "observation_update.covered_valid_area",
            "observation_update.coverage_rate",
            "observation_update.coverage_rate_delta",
            "observation_update.total_valid_cell_count",
            "observation_update.covered_valid_cell_count",
            "top_sequences.delta_c",
            "top_sequences.value_coverage",
            "top_sequences.risk",
            "top_sequences.segment_path_costs",
            "top_sequences.unreachable_reasons",
            "top_sequences.risk_reasons",
        ],
    }


def build_path_planner_sidecar(
    grid: GridMap,
    constraints: ConstraintResult,
    *,
    scenario_id: str,
    map_source: dict[str, Any] | None = None,
    platform: str | None = None,
    platform_parameters: Any | None = None,
    safety_margin_m: float = 0.0,
    include_terrain_layers: bool = True,
) -> dict[str, Any]:
    """构建 path-planner-request/v1 可直接消费的地图 sidecar。

    该 sidecar 不改变 model-explorer-contract/v1；它补充完整代价图和硬约束掩膜，
    供 model-explorer 在路径评估阶段传给 path-planner。
    """

    cost = np.asarray(grid.require_layer("cost"), dtype=float)
    passable_mask = np.asarray(constraints.passable_mask, dtype=bool)
    if cost.shape != grid.shape:
        raise ValueError("cost layer shape must match grid shape")
    if passable_mask.shape != grid.shape:
        raise ValueError("passable_mask shape must match grid shape")

    payload: dict[str, Any] = {
        "schema_version": "path-planner-sidecar/v1",
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "resolution": grid.resolution,
            "frame_id": grid.frame_id,
            "origin": list(grid.origin),
        },
        "cost": cost.tolist(),
        "passable_mask": passable_mask.tolist(),
        "metadata": {
            "source": "dev-platform-constraints",
            "scenario_id": scenario_id,
            "map_source": dict(map_source or {}),
            "platform": platform,
            "blocked_count": int(np.count_nonzero(~passable_mask)),
            "passable_ratio": float(np.mean(passable_mask)) if passable_mask.size else 0.0,
            "platform_goal_admissibility": _platform_goal_admissibility_payload(
                passable_mask,
                resolution=grid.resolution,
                platform_parameters=platform_parameters,
                safety_margin_m=safety_margin_m,
            ),
        },
    }
    if include_terrain_layers:
        payload["terrain_layers"] = {
            name: np.asarray(grid.layers[name]).tolist()
            for name in (
                "slope",
                "roughness",
                "illumination",
                "confidence",
                "obstacle",
                "obstacle_height",
                "traversability",
            )
            if name in grid.layers
        }
    return payload
