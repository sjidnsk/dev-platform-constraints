from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CandidateGoal:
    cell: tuple[int, int]
    information_gain: float
    value: float
    confidence_gain: float
    risk: float
    path_cost: float
    energy_cost: float = 0.0
    reachable: bool = True


@dataclass(frozen=True)
class ExplorationWeights:
    information_gain: float = 0.25
    value: float = 0.20
    confidence_gain: float = 0.25
    risk: float = 0.15
    path_cost: float = 0.10
    energy_cost: float = 0.05
    unreachable_penalty: float = 1.0


@dataclass(frozen=True)
class ScoredGoal:
    candidate: CandidateGoal
    utility: float
    normalized_terms: dict[str, float]


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(max(values), 0.0)
    if maximum <= 0.0:
        return tuple(0.0 for _ in values)
    return tuple(max(value, 0.0) / maximum for value in values)


def rank_exploration_goals(
    candidates: Iterable[CandidateGoal],
    weights: ExplorationWeights | None = None,
) -> tuple[ScoredGoal, ...]:
    """对离散候选观测目标按归一化探索效用排序。"""

    weights = weights or ExplorationWeights()
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        return tuple()

    normalized = {
        "information_gain": _normalize(tuple(goal.information_gain for goal in candidate_tuple)),
        "value": _normalize(tuple(goal.value for goal in candidate_tuple)),
        "confidence_gain": _normalize(tuple(goal.confidence_gain for goal in candidate_tuple)),
        "risk": _normalize(tuple(goal.risk for goal in candidate_tuple)),
        "path_cost": _normalize(tuple(goal.path_cost for goal in candidate_tuple)),
        "energy_cost": _normalize(tuple(goal.energy_cost for goal in candidate_tuple)),
    }

    scored: list[ScoredGoal] = []
    for index, goal in enumerate(candidate_tuple):
        terms = {name: values[index] for name, values in normalized.items()}
        utility = (
            weights.information_gain * terms["information_gain"]
            + weights.value * terms["value"]
            + weights.confidence_gain * terms["confidence_gain"]
            - weights.risk * terms["risk"]
            - weights.path_cost * terms["path_cost"]
            - weights.energy_cost * terms["energy_cost"]
        )
        if not goal.reachable:
            utility -= weights.unreachable_penalty
        scored.append(ScoredGoal(candidate=goal, utility=float(utility), normalized_terms=terms))

    return tuple(sorted(scored, key=lambda item: item.utility, reverse=True))
