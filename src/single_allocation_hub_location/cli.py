"""Command-line entry point for a bounded CAB25 solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_matrix_pair
from .model import build_hub_model
from .scenarios import generate_flow_scenarios
from .solution import solve_hub_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the CAB25 single-allocation model")
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=60.0)
    args = parser.parse_args()

    matrices = load_matrix_pair(
        Path("data/raw/wij_CAB25.xlsx"), Path("data/raw/C-CAB25.csv")
    )
    scenarios = generate_flow_scenarios(
        matrices.flow, scenario_count=args.scenarios, seed=args.seed
    )
    model = build_hub_model(
        matrices.distance,
        scenarios.flows,
        scenarios.probabilities,
        p=args.p,
        alpha=args.alpha,
        beta=args.beta,
    )
    solution = solve_hub_model(model, time_limit=args.time_limit)
    print(
        json.dumps(
            {
                "status": solution.status,
                "proven_optimal": solution.proven_optimal,
                "objective": solution.objective,
                "hubs": solution.hubs,
                "assignments": solution.assignments,
            }
        )
    )
