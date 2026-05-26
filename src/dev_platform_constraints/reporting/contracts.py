from __future__ import annotations

from typing import Any

from ..core import ValidationReport
from ..core.layers import GridMap


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
