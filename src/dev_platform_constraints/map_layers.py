from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LayerMetadata:
    source_id: str
    timestamp: float
    resolution: float
    frame_id: str
    unit: str
    valid_ratio: float | None = None
    source_kind: str | None = None


@dataclass
class GridMap:
    resolution: float
    origin: tuple[float, float]
    width: int
    height: int
    frame_id: str
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, LayerMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def add_layer(self, name: str, data: np.ndarray, metadata: LayerMetadata) -> None:
        array = np.asarray(data)
        if array.shape != self.shape:
            raise ValueError(f"layer {name!r} shape {array.shape} does not match grid shape {self.shape}")
        if metadata.resolution != self.resolution:
            raise ValueError(f"layer {name!r} resolution does not match grid resolution")
        if metadata.frame_id != self.frame_id:
            raise ValueError(f"layer {name!r} frame_id does not match grid frame_id")
        if metadata.valid_ratio is None:
            metadata = replace(metadata, valid_ratio=self._valid_ratio(name, array))
        self.layers[name] = array.copy()
        self.metadata[name] = metadata

    def _valid_ratio(self, name: str, array: np.ndarray) -> float:
        if name == "valid_mask":
            return float(np.mean(array.astype(bool)))
        finite = np.isfinite(np.asarray(array, dtype=float))
        if "valid_mask" in self.layers and self.layers["valid_mask"].shape == self.shape:
            finite = finite & self.layers["valid_mask"].astype(bool, copy=False)
        return float(np.count_nonzero(finite) / finite.size)

    def has_layer(self, name: str) -> bool:
        return name in self.layers

    def require_layer(self, name: str) -> np.ndarray:
        try:
            return self.layers[name]
        except KeyError as exc:
            raise KeyError(f"required layer {name!r} is missing") from exc

    def layer_metadata(self, name: str) -> LayerMetadata:
        try:
            return self.metadata[name]
        except KeyError as exc:
            raise KeyError(f"metadata for layer {name!r} is missing") from exc

    def copy(self) -> "GridMap":
        return GridMap(
            resolution=self.resolution,
            origin=self.origin,
            width=self.width,
            height=self.height,
            frame_id=self.frame_id,
            layers={name: layer.copy() for name, layer in self.layers.items()},
            metadata=dict(self.metadata),
        )


def derived_metadata(
    source_layer: str,
    source_metadata: LayerMetadata,
    unit: str,
    source_kind: str = "derived",
) -> LayerMetadata:
    return LayerMetadata(
        source_id=f"{source_kind}:{source_layer}",
        timestamp=source_metadata.timestamp,
        resolution=source_metadata.resolution,
        frame_id=source_metadata.frame_id,
        unit=unit,
        source_kind=source_kind,
    )


def metadata_for_generated_layer(
    name: str,
    resolution: float,
    frame_id: str,
    unit: str = "unitless",
    timestamp: float = 0.0,
    source_kind: str = "generated",
) -> LayerMetadata:
    return LayerMetadata(
        source_id=f"{source_kind}:{name}",
        timestamp=timestamp,
        resolution=resolution,
        frame_id=frame_id,
        unit=unit,
        source_kind=source_kind,
    )


def ensure_metadata_map(metadata: Mapping[str, LayerMetadata]) -> dict[str, LayerMetadata]:
    return dict(metadata)
