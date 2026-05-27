from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import ConfidenceComponent


@dataclass(frozen=True)
class BayesianStateLayer:
    name: str
    values: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class CategoricalBayesianStateLayer:
    name: str
    categories: tuple[str, ...]
    probabilities: np.ndarray
    valid_mask: np.ndarray


TERRAIN_CATEGORIES = ("safe_regolith", "rough", "obstacle", "shadow_risk")


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-9, 1.0 - 1e-9)


def _normalize_categories(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    totals = np.sum(clipped, axis=-1, keepdims=True)
    return clipped / totals


def update_obstacle_posterior(
    prior: np.ndarray,
    observation: np.ndarray,
    observation_confidence: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> BayesianStateLayer:
    """用简化 log-odds 形式更新障碍概率后验。"""

    return _update_probability_posterior("obstacle", prior, observation, observation_confidence, valid_mask)


def update_traversability_posterior(
    prior: np.ndarray,
    observation: np.ndarray,
    observation_confidence: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> BayesianStateLayer:
    """用简化 log-odds 形式更新通行概率后验。"""

    return _update_probability_posterior("traversable", prior, observation, observation_confidence, valid_mask)


def update_categorical_posterior(
    prior: np.ndarray,
    observation_likelihood: np.ndarray,
    observation_confidence: np.ndarray,
    *,
    categories: tuple[str, ...] = TERRAIN_CATEGORIES,
    valid_mask: np.ndarray | None = None,
    name: str = "terrain",
) -> CategoricalBayesianStateLayer:
    """用离散地形类别似然更新后验，保持类别状态与可信度派生分离。"""

    prior_array = _normalize_categories(prior)
    likelihood = _normalize_categories(observation_likelihood)
    quality = np.clip(np.asarray(observation_confidence, dtype=float), 0.0, 1.0)
    if prior_array.shape != likelihood.shape:
        raise ValueError("prior and observation_likelihood must share the same shape")
    if prior_array.ndim < 2 or prior_array.shape[-1] != len(categories):
        raise ValueError("last probability dimension must match categories")
    if quality.shape != prior_array.shape[:-1]:
        raise ValueError("observation_confidence shape must match probability grid shape")

    if valid_mask is None:
        valid = np.ones(prior_array.shape[:-1], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != prior_array.shape[:-1]:
            raise ValueError("valid_mask shape must match probability grid shape")

    posterior = prior_array.copy()
    weighted_likelihood = np.power(likelihood, quality[..., None])
    updated = _normalize_categories(prior_array * weighted_likelihood)
    posterior[valid] = updated[valid]
    return CategoricalBayesianStateLayer(
        name=name,
        categories=tuple(categories),
        probabilities=posterior,
        valid_mask=valid,
    )


def _update_probability_posterior(
    name: str,
    prior: np.ndarray,
    observation: np.ndarray,
    observation_confidence: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> BayesianStateLayer:
    """更新一个 0 到 1 概率状态层，保持状态估计与可信度派生分离。"""

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
    return BayesianStateLayer(name=name, values=np.clip(posterior, 0.0, 1.0), valid_mask=valid)


def derive_confidence_from_posterior(posterior: BayesianStateLayer) -> ConfidenceComponent:
    """从后验概率熵派生模型可信度分量。"""

    values = _clip_probability(posterior.values)
    entropy = -(values * np.log2(values) + (1.0 - values) * np.log2(1.0 - values))
    confidence = np.clip(1.0 - entropy, 0.0, 1.0)
    valid = np.asarray(posterior.valid_mask, dtype=bool)
    return ConfidenceComponent(name="model", values=np.where(valid, confidence, 0.0), valid_mask=valid)


def derive_confidence_from_categorical_posterior(posterior: CategoricalBayesianStateLayer) -> ConfidenceComponent:
    """从离散类别后验的归一化熵派生模型可信度分量。"""

    probabilities = _normalize_categories(posterior.probabilities)
    category_count = probabilities.shape[-1]
    if category_count <= 1:
        confidence = np.ones(probabilities.shape[:-1], dtype=float)
    else:
        entropy = -np.sum(probabilities * np.log2(probabilities), axis=-1) / np.log2(category_count)
        confidence = np.clip(1.0 - entropy, 0.0, 1.0)
    valid = np.asarray(posterior.valid_mask, dtype=bool)
    return ConfidenceComponent(name="model", values=np.where(valid, confidence, 0.0), valid_mask=valid)
