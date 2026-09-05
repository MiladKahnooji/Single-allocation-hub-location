"""MILP construction for risk-averse single-allocation hub location."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp


@dataclass(frozen=True)
class BuiltHubModel:
    """A PuLP model and the variables needed to extract a solution."""

    problem: pulp.LpProblem
    hub: dict[int, pulp.LpVariable]
    assignment: dict[tuple[int, int], pulp.LpVariable]
    distance: np.ndarray
    scenario_flows: np.ndarray
    probabilities: np.ndarray
    alpha: float
    beta: float


def build_hub_model(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    p: int,
    alpha: float,
    beta: float,
) -> BuiltHubModel:
    """Build the single-allocation MILP with a conditional beta-mean objective."""
    distances, flows, weights = _validated_inputs(
        distance, scenario_flows, probabilities, p, alpha, beta
    )
    scenario_count, size, _ = flows.shape
    nodes = range(size)
    scenarios = range(scenario_count)

    problem = pulp.LpProblem("risk_averse_single_allocation_hub_location", pulp.LpMinimize)
    hub = {k: problem.add_variable(f"hub_{k}", cat="Binary") for k in nodes}
    assignment = {
        (i, k): problem.add_variable(f"assign_{i}_{k}", cat="Binary")
        for i in nodes
        for k in nodes
    }
    routed = {
        (s, i, k, m): problem.add_variable(f"route_{s}_{i}_{k}_{m}", lowBound=0)
        for s in scenarios
        for i in nodes
        for k in nodes
        for m in nodes
    }
    eta = problem.add_variable("risk_threshold", lowBound=0)
    excess = {
        s: problem.add_variable(f"risk_excess_{s}", lowBound=0) for s in scenarios
    }

    problem += pulp.lpSum(hub.values()) == p, "select_exactly_p_hubs"
    for i in nodes:
        problem += pulp.lpSum(assignment[i, k] for k in nodes) == 1, f"assign_node_{i}"
        for k in nodes:
            problem += assignment[i, k] <= hub[k], f"assign_{i}_only_to_open_{k}"
        problem += assignment[i, i] == hub[i], f"opened_hub_{i}_serves_itself"

    scenario_costs: dict[int, pulp.LpAffineExpression] = {}
    for s in scenarios:
        origin_totals = flows[s].sum(axis=1)
        destination_totals = flows[s].sum(axis=0)
        for i in nodes:
            for k in nodes:
                problem += (
                    pulp.lpSum(routed[s, i, k, m] for m in nodes)
                    == float(origin_totals[i]) * assignment[i, k]
                ), f"origin_flow_{s}_{i}_{k}"
            for m in nodes:
                problem += (
                    pulp.lpSum(routed[s, i, k, m] for k in nodes)
                    == pulp.lpSum(float(flows[s, i, j]) * assignment[j, m] for j in nodes)
                ), f"destination_flow_{s}_{i}_{m}"

        collection = pulp.lpSum(
            float(origin_totals[i] * distances[i, k]) * assignment[i, k]
            for i in nodes
            for k in nodes
        )
        transfer = pulp.lpSum(
            float(alpha * distances[k, m]) * routed[s, i, k, m]
            for i in nodes
            for k in nodes
            for m in nodes
        )
        distribution = pulp.lpSum(
            float(destination_totals[j] * distances[m, j]) * assignment[j, m]
            for j in nodes
            for m in nodes
        )
        scenario_costs[s] = collection + transfer + distribution
        problem += excess[s] >= scenario_costs[s] - eta, f"risk_tail_{s}"

    problem += eta + pulp.lpSum(
        float(weights[s] / (1.0 - beta)) * excess[s] for s in scenarios
    )
    return BuiltHubModel(
        problem=problem,
        hub=hub,
        assignment=assignment,
        distance=distances,
        scenario_flows=flows,
        probabilities=weights,
        alpha=alpha,
        beta=beta,
    )


def _validated_inputs(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    p: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.asarray(distance, dtype=float)
    flows = np.asarray(scenario_flows, dtype=float)
    weights = np.asarray(probabilities, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance must be a square matrix")
    size = distances.shape[0]
    if flows.ndim != 3 or flows.shape[1:] != (size, size) or flows.shape[0] == 0:
        raise ValueError("scenario flows must have shape (scenarios, nodes, nodes)")
    if weights.shape != (flows.shape[0],):
        raise ValueError("probabilities must match the scenario count")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distance must contain finite, nonnegative values")
    if not np.isfinite(flows).all() or (flows < 0).any():
        raise ValueError("scenario flows must contain finite, nonnegative values")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("probabilities must be nonnegative and sum to 1")
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or not 1 <= p <= size:
        raise ValueError("p must be an integer between 1 and the number of nodes")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not 0 <= beta < 1:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    return distances.copy(), flows.copy(), weights.copy()
