"""MILP construction for risk-averse single-allocation hub location."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from .risk import tail_scenario_count


@dataclass(frozen=True)
class BuiltHubModel:
    """A PuLP model and the variables needed to extract a solution."""

    problem: pulp.LpProblem
    assignment: dict[tuple[int, int], pulp.LpVariable]
    route_selection: dict[tuple[int, int, int, int, int], pulp.LpVariable]
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
    """Build the article-aligned binary-route MILP for tiny instances only.

    ``assignment[k, k]`` is the hub-opening decision and ``assignment[i, k]``
    is the first-stage single allocation.  ``route_selection[s, i, j, k, m]``
    is binary and selects the unique route through assigned hubs ``k`` and
    ``m`` for OD pair ``(i, j)`` in scenario ``s``.
    """
    distances, flows, weights = _validated_inputs(
        distance, scenario_flows, probabilities, p, alpha, beta
    )
    scenario_count, size, _ = flows.shape
    if size > 6:
        raise ValueError(
            "the binary route-selection exact model is limited to at most 6 nodes; "
            "use a heuristic or GVNS/DL-GVNS for CAB/AP datasets"
        )
    nodes = range(size)
    scenarios = range(scenario_count)

    problem = pulp.LpProblem("risk_averse_single_allocation_hub_location", pulp.LpMinimize)
    assignment = {
        (i, k): problem.add_variable(f"assign_{i}_{k}", cat="Binary")
        for i in nodes
        for k in nodes
    }
    route_selection = {
        (s, i, j, k, m): problem.add_variable(
            f"route_select_{s}_{i}_{j}_{k}_{m}", cat="Binary"
        )
        for s in scenarios
        for i in nodes
        for j in nodes
        for k in nodes
        for m in nodes
    }
    eta = problem.add_variable("risk_threshold", lowBound=0)
    excess = {
        s: problem.add_variable(f"risk_excess_{s}", lowBound=0) for s in scenarios
    }

    problem += (
        pulp.lpSum(assignment[k, k] for k in nodes) == p,
        "select_exactly_p_hubs_from_assignment_diagonal",
    )
    for i in nodes:
        problem += pulp.lpSum(assignment[i, k] for k in nodes) == 1, f"assign_node_{i}"
        for k in nodes:
            problem += (
                assignment[i, k] <= assignment[k, k],
                f"assign_{i}_only_to_open_diagonal_{k}",
            )

    scenario_costs: dict[int, pulp.LpAffineExpression] = {}
    for s in scenarios:
        for i in nodes:
            for j in nodes:
                problem += (
                    pulp.lpSum(
                        route_selection[s, i, j, k, m]
                        for k in nodes
                        for m in nodes
                    )
                    == 1,
                    f"select_one_route_{s}_{i}_{j}",
                )
                for k in nodes:
                    for m in nodes:
                        selected_route = route_selection[s, i, j, k, m]
                        problem += (
                            selected_route <= assignment[i, k],
                            f"route_origin_link_{s}_{i}_{j}_{k}_{m}",
                        )
                        problem += (
                            selected_route <= assignment[j, m],
                            f"route_destination_link_{s}_{i}_{j}_{k}_{m}",
                        )
                        problem += (
                            selected_route >= assignment[i, k] + assignment[j, m] - 1,
                            f"route_assignment_product_{s}_{i}_{j}_{k}_{m}",
                        )
        scenario_costs[s] = pulp.lpSum(
            float(flows[s, i, j])
            * (distances[i, k] + alpha * distances[k, m] + distances[m, j])
            * route_selection[s, i, j, k, m]
            for i in nodes
            for j in nodes
            for k in nodes
            for m in nodes
        )
        problem += excess[s] >= scenario_costs[s] - eta, f"risk_tail_{s}"

    if np.allclose(weights, 1.0 / scenario_count):
        # The paper's finite equally likely scenario form: average the worst
        # K = ceil(beta * S) scenario costs.  This remains meaningful when
        # beta * S is not an integer.
        tail_count = tail_scenario_count(scenario_count, beta)
        problem += eta + pulp.lpSum(excess[s] for s in scenarios) / tail_count
    else:
        # Preserve the generalized explicit-probability conditional beta-mean.
        problem += eta + pulp.lpSum(
            float(weights[s] / beta) * excess[s] for s in scenarios
        )
    return BuiltHubModel(
        problem=problem,
        assignment=assignment,
        route_selection=route_selection,
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
    if not 0 < beta <= 1:
        raise ValueError("beta must satisfy 0 < beta <= 1")
    return distances.copy(), flows.copy(), weights.copy()
