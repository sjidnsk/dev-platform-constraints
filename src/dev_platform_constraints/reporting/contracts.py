from __future__ import annotations

from dataclasses import asdict, is_dataclass
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
                "coverage_area": float(goal.candidate.coverage_area),
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
            "top_sequences.delta_c",
            "top_sequences.value_coverage",
            "top_sequences.risk",
            "top_sequences.segment_path_costs",
            "top_sequences.unreachable_reasons",
            "top_sequences.risk_reasons",
        ],
    }
