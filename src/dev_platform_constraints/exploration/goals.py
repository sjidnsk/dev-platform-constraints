from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import atan2, degrees, hypot
from typing import Iterable

import numpy as np

from ..confidence import compute_observation_model
from ..core.layers import GridMap
from ..mapping.constraints import ConstraintResult
from ..path_planning import astar_path
from ..platforms import PlatformParameters


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


@dataclass(frozen=True)
class GoalSequenceEvaluation:
    goals: tuple[CandidateGoal, ...]
    utility: float
    delta_c: float
    value_coverage: float
    risk: float
    path_cost: float
    reachable: bool


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(max(values), 0.0)
    if maximum <= 0.0:
        return tuple(0.0 for _ in values)
    return tuple(max(value, 0.0) / maximum for value in values)


def _layer_or_default(grid: GridMap, name: str, default: float) -> np.ndarray:
    if name in grid.layers:
        return np.nan_to_num(np.asarray(grid.layers[name], dtype=float), nan=default)
    return np.full(grid.shape, default, dtype=float)


def _candidate_risk(grid: GridMap, platform: PlatformParameters) -> np.ndarray:
    slope = _layer_or_default(grid, "slope", platform.max_slope_deg)
    roughness = _layer_or_default(grid, "roughness", 1.0)
    obstacle = _layer_or_default(grid, "obstacle", 1.0)
    illumination = _layer_or_default(grid, "illumination", 0.0)
    confidence = _layer_or_default(grid, "confidence", 0.0)
    slope_risk = np.clip(slope / max(platform.max_slope_deg, 1e-9), 0.0, 1.0)
    roughness_risk = np.clip(roughness, 0.0, 1.0)
    obstacle_risk = np.clip(obstacle, 0.0, 1.0)
    illumination_risk = np.clip(1.0 - illumination, 0.0, 1.0)
    confidence_risk = np.clip(1.0 - confidence, 0.0, 1.0)
    return np.clip(0.30 * slope_risk + 0.20 * roughness_risk + 0.25 * obstacle_risk + 0.15 * illumination_risk + 0.10 * confidence_risk, 0.0, 1.0)


def _frontier_score(valid: np.ndarray, passable: np.ndarray) -> np.ndarray:
    blocked_or_invalid = (~valid) | (~passable)
    frontier = np.zeros(valid.shape, dtype=float)
    height, width = valid.shape
    for y in range(height):
        for x in range(width):
            y0, y1 = max(0, y - 1), min(height, y + 2)
            x0, x1 = max(0, x - 1), min(width, x + 2)
            if blocked_or_invalid[y0:y1, x0:x1].any():
                frontier[y, x] = 1.0
    return frontier


def _heading_from_start(start: tuple[int, int], cell: tuple[int, int]) -> float:
    dx = cell[0] - start[0]
    dy = cell[1] - start[1]
    if dx == 0 and dy == 0:
        return 0.0
    return degrees(atan2(dy, dx))


def _footprint_gain_layers(
    grid: GridMap,
    start: tuple[int, int],
    platform: PlatformParameters,
    valid: np.ndarray,
    confidence_gain: np.ndarray,
    value: np.ndarray,
    frontier: np.ndarray,
    lookahead_steps: int,
    use_simple_occlusion: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    footprint_information = np.zeros(grid.shape, dtype=float)
    footprint_value = np.zeros(grid.shape, dtype=float)
    footprint_confidence_gain = np.zeros(grid.shape, dtype=float)

    def footprint_scores(cell: tuple[int, int], heading_reference: tuple[int, int]) -> tuple[float, float]:
        model = compute_observation_model(
            grid,
            platform,
            cell,
            _heading_from_start(heading_reference, cell),
            use_simple_occlusion=use_simple_occlusion,
        )
        footprint_weight = np.asarray(model.quality_layer, dtype=float) * valid
        if not np.any(footprint_weight > 0.0):
            x, y = cell
            return float(confidence_gain[y, x]), float(value[y, x])
        return (
            max(float(confidence_gain[cell[1], cell[0]]), float(np.max(confidence_gain * footprint_weight))),
            max(float(value[cell[1], cell[0]]), float(np.max(value * footprint_weight))),
        )

    for y in range(grid.height):
        for x in range(grid.width):
            if not valid[y, x]:
                continue
            cell = (x, y)
            direct_confidence_gain, direct_value = footprint_scores(cell, start)
            if lookahead_steps >= 2:
                model = compute_observation_model(
                    grid,
                    platform,
                    cell,
                    _heading_from_start(start, cell),
                    use_simple_occlusion=use_simple_occlusion,
                )
                downstream_cells = np.argwhere((model.quality_layer > 0.0) & valid)
                for downstream_y, downstream_x in downstream_cells:
                    downstream_cell = (int(downstream_x), int(downstream_y))
                    if downstream_cell == cell:
                        continue
                    downstream_confidence_gain, downstream_value = footprint_scores(downstream_cell, cell)
                    direct_confidence_gain = max(direct_confidence_gain, downstream_confidence_gain)
                    direct_value = max(direct_value, downstream_value)
            footprint_confidence_gain[y, x] = direct_confidence_gain
            footprint_value[y, x] = direct_value
            footprint_information[y, x] = float(
                np.clip(0.7 * footprint_confidence_gain[y, x] + 0.3 * frontier[y, x], 0.0, 1.0)
            )
    return footprint_information, footprint_value, footprint_confidence_gain


def generate_exploration_candidates(
    grid: GridMap,
    constraints: ConstraintResult,
    start: tuple[int, int],
    platform: PlatformParameters,
    *,
    max_candidates: int = 8,
    lookahead_steps: int = 1,
    use_simple_occlusion: bool = False,
) -> tuple[CandidateGoal, ...]:
    """从地图层生成离散探索候选点，供效用排序复用。"""

    if max_candidates <= 0:
        return tuple()
    if lookahead_steps <= 0:
        raise ValueError("lookahead_steps must be positive")
    sx, sy = start
    if not (0 <= sx < grid.width and 0 <= sy < grid.height):
        raise ValueError("start must be inside the grid")

    valid = grid.layers.get("valid_mask", np.ones(grid.shape, dtype=bool)).astype(bool, copy=False)
    passable = np.asarray(constraints.passable_mask, dtype=bool)
    if passable.shape != grid.shape:
        raise ValueError("constraints passable_mask shape must match grid shape")

    confidence = np.clip(_layer_or_default(grid, "confidence", 0.0), 0.0, 1.0)
    value = np.clip(_layer_or_default(grid, "value", 0.0), 0.0, 1.0)
    confidence_gain = np.clip(1.0 - confidence, 0.0, 1.0)
    risk = _candidate_risk(grid, platform)
    frontier = _frontier_score(valid, passable)
    footprint_information, footprint_value, footprint_confidence_gain = _footprint_gain_layers(
        grid,
        start,
        platform,
        valid,
        confidence_gain,
        value,
        frontier,
        lookahead_steps,
        use_simple_occlusion,
    )
    seed_score = np.where(
        valid,
        0.35 * footprint_confidence_gain + 0.30 * footprint_value + 0.15 * frontier + 0.10 * (1.0 - risk) + 0.10 * confidence_gain,
        -1.0,
    )

    flat_order = np.argsort(seed_score.ravel())[::-1]
    candidates: list[CandidateGoal] = []
    seen: set[tuple[int, int]] = set()
    cost = grid.layers.get("cost")
    for flat_index in flat_order:
        if len(candidates) >= max_candidates:
            break
        y, x = np.unravel_index(int(flat_index), grid.shape)
        cell = (int(x), int(y))
        if cell in seen or seed_score[y, x] <= 0.0:
            continue
        seen.add(cell)
        reachable = False
        path_cost = hypot(x - sx, y - sy) * grid.resolution
        if cost is not None and passable[sy, sx] and passable[y, x]:
            plan = astar_path(np.asarray(cost, dtype=float), passable, start, cell, grid.resolution)
            reachable = plan.reachable
            if plan.reachable:
                path_cost = float(plan.total_cost)
            else:
                path_cost += grid.width * grid.height * grid.resolution
        else:
            path_cost += grid.width * grid.height * grid.resolution

        candidates.append(
            CandidateGoal(
                cell=cell,
                information_gain=float(footprint_information[y, x]),
                value=float(footprint_value[y, x]),
                confidence_gain=float(footprint_confidence_gain[y, x]),
                risk=float(risk[y, x]),
                path_cost=float(path_cost),
                energy_cost=float(path_cost * (1.0 + risk[y, x])),
                reachable=reachable,
            )
        )
    return tuple(candidates)


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


def evaluate_goal_sequences(
    candidates: Iterable[CandidateGoal],
    weights: ExplorationWeights | None = None,
    *,
    depth: int = 3,
    beam_width: int = 5,
) -> tuple[GoalSequenceEvaluation, ...]:
    """离线评估多步候选目标序列，不执行闭环重规划。"""

    if depth <= 0:
        raise ValueError("depth must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    weights = weights or ExplorationWeights()
    ranked_candidates = tuple(scored.candidate for scored in rank_exploration_goals(candidates, weights))[:beam_width]
    if not ranked_candidates:
        return tuple()

    raw_sequences: list[dict[str, object]] = []
    max_depth = min(depth, len(ranked_candidates))
    for length in range(1, max_depth + 1):
        for sequence in permutations(ranked_candidates, length):
            risk = float(max(goal.risk for goal in sequence))
            raw_sequences.append(
                {
                    "goals": sequence,
                    "information_gain": float(sum(goal.information_gain for goal in sequence)),
                    "delta_c": float(sum(goal.confidence_gain for goal in sequence)),
                    "value_coverage": float(sum(goal.value for goal in sequence)),
                    "risk": risk,
                    "path_cost": float(sum(goal.path_cost for goal in sequence)),
                    "energy_cost": float(sum(goal.energy_cost for goal in sequence)),
                    "reachable": all(goal.reachable for goal in sequence),
                }
            )

    normalized = {
        key: _normalize(tuple(float(item[key]) for item in raw_sequences))
        for key in ("information_gain", "delta_c", "value_coverage", "risk", "path_cost", "energy_cost")
    }
    evaluations: list[GoalSequenceEvaluation] = []
    for index, item in enumerate(raw_sequences):
        risk_term = normalized["risk"][index]
        utility = (
            weights.information_gain * normalized["information_gain"][index]
            + weights.confidence_gain * normalized["delta_c"][index]
            + weights.value * normalized["value_coverage"][index]
            - weights.risk * risk_term
            - weights.path_cost * normalized["path_cost"][index]
            - weights.energy_cost * normalized["energy_cost"][index]
        )
        if float(item["risk"]) >= 0.70:
            utility -= weights.unreachable_penalty * float(item["risk"])
        if not bool(item["reachable"]):
            utility -= weights.unreachable_penalty
        evaluations.append(
            GoalSequenceEvaluation(
                goals=tuple(item["goals"]),
                utility=float(utility),
                delta_c=float(item["delta_c"]),
                value_coverage=float(item["value_coverage"]),
                risk=float(item["risk"]),
                path_cost=float(item["path_cost"]),
                reachable=bool(item["reachable"]),
            )
        )
    return tuple(sorted(evaluations, key=lambda item: item.utility, reverse=True))
