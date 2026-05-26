"""核心地图数据结构与契约。"""

from .contracts import CORE_LAYERS, ValidationIssue, ValidationReport, validate_grid_map
from .layers import GridMap, LayerMetadata, derived_metadata, ensure_metadata_map, metadata_for_generated_layer

__all__ = [
    "CORE_LAYERS",
    "GridMap",
    "LayerMetadata",
    "ValidationIssue",
    "ValidationReport",
    "derived_metadata",
    "ensure_metadata_map",
    "metadata_for_generated_layer",
    "validate_grid_map",
]
