"""Solve built hub models and represent their incumbent solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from .evaluation import evaluate_scenario_costs
from .model import BuiltHubModel
from .risk import conditional_beta_mean, equal_probability_beta_mean


@dataclass(frozen=True)
class HubSolution:
    """A solver result with explicit hubs, assignments, and evaluated costs."""

    status: str
    proven_optimal: bool
    hubs: tuple[int, ...]
    assignments: tuple[int, ...]
    scenario_costs: np.ndarray
    objective: float | None


def solve_hub_model(model: BuiltHubModel, time_limit: float | None = None) -> HubSolution:
    """Solve a built model with the open-source CBC solver bundled by PuLP."""
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    solver = pulp.COIN_CMD(
        path=pulp.apis.PULP_CBC_CMD.pulp_cbc_path,
        msg=False,
        timeLimit=time_limit,
        threads=1,
        options=["randomSeed 0"],
    )
    model.problem.solve(solver)
    status = pulp.LpSolution[model.problem.sol_status]
    proven_optimal = model.problem.sol_status == pulp.LpSolutionOptimal
    size = model.distance.shape[0]
    hubs = tuple(
        k for k in range(size) if (pulp.value(model.assignment[k, k]) or 0.0) > 0.5
    )
    assignments = tuple(
        max(range(size), key=lambda k: pulp.value(model.assignment[i, k]) or 0.0)
        for i in range(size)
    )
    if len(hubs) == 0 or any(assigned not in hubs for assigned in assignments):
        return HubSolution(
            status=status,
            proven_optimal=False,
            hubs=(),
            assignments=(),
            scenario_costs=np.array([], dtype=float),
            objective=None,
        )
    costs = evaluate_scenario_costs(
        assignments, model.distance, model.scenario_flows, model.alpha
    )
    if np.allclose(model.probabilities, 1.0 / model.probabilities.size):
        objective = equal_probability_beta_mean(costs, model.beta)
    else:
        objective = conditional_beta_mean(costs, model.probabilities, model.beta)
    return HubSolution(
        status=status,
        proven_optimal=proven_optimal,
        hubs=hubs,
        assignments=assignments,
        scenario_costs=costs,
        objective=objective,
    )
