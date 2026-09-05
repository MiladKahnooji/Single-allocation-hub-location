"""Evaluate fixed single-allocation hub solutions."""

from __future__ import annotations

import numpy as np

from .risk import conditional_beta_mean


def evaluate_scenario_costs(
    assignments: tuple[int, ...],
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Calculate scenario costs for fixed node-to-hub assignments."""
    distances = np.asarray(distance, dtype=float)
    flows = np.asarray(scenario_flows, dtype=float)
    assigned = np.asarray(assignments, dtype=int)
    size = distances.shape[0]
    if distances.shape != (size, size) or flows.ndim != 3 or flows.shape[1:] != distances.shape:
        raise ValueError("distance and scenario flow dimensions must match")
    if assigned.shape != (size,) or (assigned < 0).any() or (assigned >= size).any():
        raise ValueError("assignments must contain one valid hub index per node")

    origin_legs = distances[np.arange(size), assigned]
    destination_legs = distances[assigned, np.arange(size)]
    inter_hub = distances[assigned[:, None], assigned[None, :]]
    route_costs = origin_legs[:, None] + alpha * inter_hub + destination_legs
    return np.sum(flows * route_costs, axis=(1, 2))


def evaluate_risk_objective(
    assignments: tuple[int, ...],
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Evaluate the conditional beta-mean for fixed assignments."""
    costs = evaluate_scenario_costs(assignments, distance, scenario_flows, alpha)
    return conditional_beta_mean(costs, probabilities, beta)
