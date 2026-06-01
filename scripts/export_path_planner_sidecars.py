from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.confidence import load_confidence_weights, load_terrain_likelihood_rules
from dev_platform_constraints.experiments import default_ablation_scenario_config_path, load_ablation_scenarios
from dev_platform_constraints.mapping import generate_costmap, generate_hard_constraints
from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters
from dev_platform_constraints.reporting import build_model_explorer_contract, build_path_planner_sidecar
from dev_platform_constraints.terrain import derive_terrain_features

from run_confidence_ablation import (
    _apply_observation_updates,
    _apply_scenario_layers,
    _apply_scenario_risk_layers,
    _build_scenario_grid,
    _resolve_terrain_likelihood_config,
    evaluate_goal_sequences,
    generate_exploration_candidates,
    rank_exploration_goals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model-explorer contracts and path-planner sidecars for semi-real scenarios."
    )
    parser.add_argument(
        "--scenario-config",
        default=str(default_ablation_scenario_config_path()),
        help="Ablation scenario config JSON. Use configs/ablation/npz_validation_scenarios.json for semi-real maps.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "path_planner_sidecars"),
        help="Directory for exported contracts, sidecars, and manifest.",
    )
    parser.add_argument(
        "--confidence-config",
        default=str(ROOT / "configs" / "confidence" / "default.json"),
        help="Confidence fusion config used before constraints/costmap export.",
    )
    parser.add_argument(
        "--terrain-likelihood-config",
        default=None,
        help="Optional terrain likelihood config; defaults to scenario config value when present.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top goals/sequences to keep in contracts.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned exports without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_config = Path(args.scenario_config)
    output_dir = Path(args.output_dir)
    terrain_config = _resolve_terrain_likelihood_config(args.terrain_likelihood_config, scenario_config)
    terrain_rules = load_terrain_likelihood_rules(terrain_config) if terrain_config is not None else None
    scenarios = load_ablation_scenarios(scenario_config)
    summary = {
        "scenario_config": str(scenario_config),
        "output_dir": str(output_dir),
        "confidence_config": str(args.confidence_config),
        "terrain_likelihood_config": None if terrain_config is None else str(terrain_config),
        "exports": [
            {
                "scenario_id": scenario.scenario_id,
                "contract": str(output_dir / f"{scenario.scenario_id}.contract.json"),
                "sidecar": str(output_dir / f"{scenario.scenario_id}.path-planner-sidecar.json"),
            }
            for scenario in scenarios
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    platform = load_platform_parameters(default_platform_config_path("yutu2"))
    weights = load_confidence_weights(args.confidence_config)

    for scenario in scenarios:
        grid = _build_scenario_grid(scenario)
        _apply_scenario_layers(grid, scenario)
        derive_terrain_features(grid, roughness_window_size=3, roughness_normalization_height=0.3)
        _apply_scenario_risk_layers(grid, scenario)
        confidence_report = _apply_observation_updates(grid, platform, scenario, weights)
        constraints = generate_hard_constraints(grid, platform)
        generate_costmap(grid, constraints, platform)
        candidates = generate_exploration_candidates(
            grid,
            constraints,
            start=scenario.start_cell,
            platform=platform,
            max_candidates=12,
            lookahead_steps=scenario.lookahead_steps,
            use_simple_occlusion=scenario.use_simple_occlusion,
        )
        scored_goals = rank_exploration_goals(candidates)[: max(args.top_k, 0)]
        goal_sequences = evaluate_goal_sequences(candidates, depth=3, beam_width=max(args.top_k, 1))[
            : max(args.top_k, 0)
        ]
        contract = build_model_explorer_contract(
            grid,
            constraints,
            scored_goals,
            goal_sequences,
            confidence_report,
        )
        sidecar = build_path_planner_sidecar(
            grid,
            constraints,
            scenario_id=scenario.scenario_id,
            map_source={
                "kind": scenario.map_source.kind,
                "path": scenario.map_source.path,
                "seed": scenario.map_source.seed,
            },
            platform="yutu2",
        )
        (output_dir / f"{scenario.scenario_id}.contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{scenario.scenario_id}.path-planner-sidecar.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
