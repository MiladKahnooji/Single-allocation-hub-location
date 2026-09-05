"""A small seeded local-search heuristic for larger instances."""

from __future__ import annotations

import time

import numpy as np

from .evaluation import evaluate_risk_objective, evaluate_scenario_costs
from .solution import HubSolution

_MAX_SWAP_ATTEMPTS = 50


def solve_hub_heuristic(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    p: int,
    alpha: float,
    beta: float,
    seed: int,
    time_limit: float,
) -> HubSolution:
    """Build and improve a feasible solution using seeded one-hub swaps."""
    distances = np.asarray(distance, dtype=float)
    flows = np.asarray(scenario_flows, dtype=float)
    weights = np.asarray(probabilities, dtype=float)
    size = distances.shape[0]
    if distances.shape != (size, size) or flows.ndim != 3 or flows.shape[1:] != distances.shape:
        raise ValueError("distance and scenario flow dimensions must match")
    if not 1 <= p <= size:
        raise ValueError("p must be between 1 and the number of nodes")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")

    rng = np.random.default_rng(seed)
    hubs = tuple(sorted(int(value) for value in rng.choice(size, size=p, replace=False)))
    assignments = _nearest_assignments(distances, hubs)
    objective = evaluate_risk_objective(assignments, distances, flows, weights, alpha, beta)
    deadline = time.perf_counter() + time_limit

    for _ in range(_MAX_SWAP_ATTEMPTS):
        if time.perf_counter() >= deadline:
            break
        closed = tuple(node for node in range(size) if node not in hubs)
        outgoing = hubs[int(rng.integers(len(hubs)))]
        incoming = closed[int(rng.integers(len(closed)))] if closed else outgoing
        candidate_hubs = tuple(sorted((set(hubs) - {outgoing}) | {incoming}))
        candidate_assignments = _nearest_assignments(distances, candidate_hubs)
        candidate_objective = evaluate_risk_objective(
            candidate_assignments, distances, flows, weights, alpha, beta
        )
        if candidate_objective < objective:
            hubs = candidate_hubs
            assignments = candidate_assignments
            objective = candidate_objective

    costs = evaluate_scenario_costs(assignments, distances, flows, alpha)
    return HubSolution(
        status="Heuristic completed",
        proven_optimal=False,
        hubs=hubs,
        assignments=assignments,
        scenario_costs=costs,
        objective=objective,
    )


def _nearest_assignments(distance: np.ndarray, hubs: tuple[int, ...]) -> tuple[int, ...]:
    hub_array = np.asarray(hubs)
    proximity = distance[:, hub_array] + distance[hub_array, :].T
    assignments = hub_array[np.argmin(proximity, axis=1)]
    assignments[hub_array] = hub_array
    return tuple(int(value) for value in assignments)
