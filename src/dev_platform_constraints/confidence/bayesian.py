from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class TerrainLikelihoodRules:
    """离散地形类别观测似然的轻量标定规则。"""

    slope_rough_deg: float = 20.0
    roughness_threshold: float = 0.6
    obstacle_threshold: float = 0.5
    shadow_threshold: float = 0.35
    base_likelihood: float = 0.02
    risk_likelihood: float = 0.98


def _validate_terrain_likelihood_rules(rules: TerrainLikelihoodRules) -> TerrainLikelihoodRules:
    if rules.slope_rough_deg <= 0.0:
        raise ValueError("slope_rough_deg must be positive")
    if rules.roughness_threshold <= 0.0:
        raise ValueError("roughness_threshold must be positive")
    if rules.obstacle_threshold <= 0.0:
        raise ValueError("obstacle_threshold must be positive")
    if rules.shadow_threshold <= 0.0:
        raise ValueError("shadow_threshold must be positive")
    if not 0.0 <= rules.base_likelihood <= 1.0 or not 0.0 <= rules.risk_likelihood <= 1.0:
        raise ValueError("terrain likelihood values must be probabilities in [0, 1]")
    if rules.base_likelihood + rules.risk_likelihood <= 0.0:
        raise ValueError("at least one terrain likelihood value must be positive")
    return rules


def load_terrain_likelihood_rules(path: str | Path) -> TerrainLikelihoodRules:
    """从 JSON 配置加载离散地形类别观测似然规则。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)
    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, dict):
        raise ValueError("terrain likelihood config must contain a rules object")

    values: dict[str, float] = {}
    for field in (
        "slope_rough_deg",
        "roughness_threshold",
        "obstacle_threshold",
        "shadow_threshold",
        "base_likelihood",
        "risk_likelihood",
    ):
        if field not in rules_raw:
            raise ValueError(f"terrain likelihood rule {field!r} is missing")
        values[field] = float(rules_raw[field])
    return _validate_terrain_likelihood_rules(TerrainLikelihoodRules(**values))


def default_terrain_likelihood_config_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / "confidence" / "terrain_likelihood_default.json"


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-9, 1.0 - 1e-9)


def _normalize_categories(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    totals = np.sum(clipped, axis=-1, keepdims=True)
    return clipped / totals


def compute_terrain_category_likelihood(
    *,
    slope: np.ndarray,
    roughness: np.ndarray,
    obstacle: np.ndarray,
    illumination: np.ndarray,
    rules: TerrainLikelihoodRules | None = None,
    valid_mask: np.ndarray | None = None,
    categories: tuple[str, ...] = TERRAIN_CATEGORIES,
) -> CategoricalBayesianStateLayer:
    """从坡度、崎岖度、障碍和光照层生成离散地形类别观测似然。"""

    if tuple(categories) != TERRAIN_CATEGORIES:
        raise ValueError("terrain categories must use the fixed public category order")
    rules = _validate_terrain_likelihood_rules(rules or TerrainLikelihoodRules())

    slope_array = np.asarray(slope, dtype=float)
    roughness_array = np.asarray(roughness, dtype=float)
    obstacle_array = np.asarray(obstacle, dtype=float)
    illumination_array = np.asarray(illumination, dtype=float)
    if not (
        slope_array.shape == roughness_array.shape == obstacle_array.shape == illumination_array.shape
    ):
        raise ValueError("slope, roughness, obstacle and illumination must share the same shape")

    if valid_mask is None:
        valid = np.ones(slope_array.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != slope_array.shape:
            raise ValueError("valid_mask shape must match terrain layer shape")

    slope_risk = np.clip(slope_array / rules.slope_rough_deg, 0.0, 1.0)
    roughness_risk = np.clip(roughness_array / rules.roughness_threshold, 0.0, 1.0)
    rough_risk = np.maximum(slope_risk, roughness_risk)
    obstacle_risk = np.clip(obstacle_array / rules.obstacle_threshold, 0.0, 1.0)
    shadow_risk = np.clip((rules.shadow_threshold - illumination_array) / rules.shadow_threshold, 0.0, 1.0)
    safe_risk = np.clip(1.0 - (rough_risk + obstacle_risk + shadow_risk), 0.0, 1.0)

    evidence = np.stack((safe_risk, rough_risk, obstacle_risk, shadow_risk), axis=-1)
    likelihood = rules.base_likelihood + rules.risk_likelihood * evidence
    probabilities = _normalize_categories(likelihood)
    return CategoricalBayesianStateLayer(
        name="terrain_likelihood",
        categories=tuple(categories),
        probabilities=np.where(valid[..., None], probabilities, 0.0),
        valid_mask=valid,
    )


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
