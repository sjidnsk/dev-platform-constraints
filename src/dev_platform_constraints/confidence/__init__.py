"""可信度指标计算、融合和局部观测更新。"""

from .metrics import (
    ConfidenceComponent,
    ConfidenceUpdateReport,
    ConfidenceWeights,
    compute_consistency_confidence,
    compute_observation_confidence,
    compute_recency_confidence,
    compute_resolution_confidence,
    fuse_confidence,
    update_confidence_from_observation,
)
from .config import default_confidence_config_path, load_confidence_weights
from .bayesian import BayesianStateLayer, derive_confidence_from_posterior, update_obstacle_posterior

__all__ = [
    "BayesianStateLayer",
    "ConfidenceComponent",
    "ConfidenceUpdateReport",
    "ConfidenceWeights",
    "default_confidence_config_path",
    "derive_confidence_from_posterior",
    "compute_consistency_confidence",
    "compute_observation_confidence",
    "compute_recency_confidence",
    "compute_resolution_confidence",
    "fuse_confidence",
    "load_confidence_weights",
    "update_obstacle_posterior",
    "update_confidence_from_observation",
]
