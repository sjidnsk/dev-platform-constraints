from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import ConfidenceComponent


@dataclass(frozen=True)
class BayesianStateLayer:
    name: str
    values: np.ndarray
    valid_mask: np.ndarray


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-9, 1.0 - 1e-9)


def update_obstacle_posterior(
    prior: np.ndarray,
    observation: np.ndarray,
    observation_confidence: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> BayesianStateLayer:
    """用简化 log-odds 形式更新障碍概率后验。"""

    prior_array = _clip_probability(prior)
    observation_array = _clip_probability(observation)
    quality = np.clip(np.asarray(observation_confidence, dtype=float), 0.0, 1.0)
    if prior_array.shape != observation_array.shape or prior_array.shape != quality.shape:
        raise ValueError("prior, observation and observation_confidence must share the same shape")

    if valid_mask is None:
        valid = np.ones(prior_array.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != prior_array.shape:
            raise ValueError("valid_mask shape must match prior shape")

    prior_logit = np.log(prior_array / (1.0 - prior_array))
    observation_logit = np.log(observation_array / (1.0 - observation_array))
    posterior = prior_array.copy()
    updated_logit = prior_logit + quality * observation_logit
    posterior[valid] = 1.0 / (1.0 + np.exp(-updated_logit[valid]))
    return BayesianStateLayer(name="obstacle", values=np.clip(posterior, 0.0, 1.0), valid_mask=valid)


def derive_confidence_from_posterior(posterior: BayesianStateLayer) -> ConfidenceComponent:
    """从后验概率熵派生模型可信度分量。"""

    values = _clip_probability(posterior.values)
    entropy = -(values * np.log2(values) + (1.0 - values) * np.log2(1.0 - values))
    confidence = np.clip(1.0 - entropy, 0.0, 1.0)
    valid = np.asarray(posterior.valid_mask, dtype=bool)
    return ConfidenceComponent(name="model", values=np.where(valid, confidence, 0.0), valid_mask=valid)
