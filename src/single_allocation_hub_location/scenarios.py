"""Reproducible stochastic flow scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAX_ZERO_TOTAL_RETRIES = 3


@dataclass(frozen=True)
class FlowScenarios:
    """Generated flow matrices and their probabilities."""

    flows: np.ndarray
    probabilities: np.ndarray


def generate_flow_scenarios(
    flow: np.ndarray, scenario_count: int = 100, seed: int | None = None
) -> FlowScenarios:
    """Generate normalized Poisson flow scenarios from a base flow matrix.

    Node multipliers are sampled independently for every scenario using a local
    ``default_rng``. A zero-total Poisson draw is retried a bounded number of
    times before generation fails clearly.
    """
    if isinstance(scenario_count, bool) or not isinstance(scenario_count, (int, np.integer)):
        raise ValueError("scenario_count must be a positive integer")
    if scenario_count <= 0:
        raise ValueError("scenario_count must be a positive integer")

    base = np.asarray(flow, dtype=float)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("flow must be a square matrix")
    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("flow must contain finite, nonnegative values")
    total = float(base.sum())
    if total <= 0:
        raise ValueError("flow must have positive total demand")

    rng = np.random.default_rng(seed)
    size = base.shape[0]
    flows = np.empty((scenario_count, size, size), dtype=float)
    for index in range(scenario_count):
        for _ in range(_MAX_ZERO_TOTAL_RETRIES + 1):
            multipliers = rng.uniform(0.5, 1.5, size=size)
            sampled = rng.poisson(base * multipliers[:, None] * multipliers[None, :])
            np.fill_diagonal(sampled, 0)
            sampled_total = sampled.sum()
            if sampled_total > 0:
                flows[index] = sampled / sampled_total
                break
        else:
            raise RuntimeError(
                f"scenario {index} had zero total flow after "
                f"{_MAX_ZERO_TOTAL_RETRIES + 1} attempts"
            )

    probabilities = np.full(scenario_count, 1.0 / scenario_count, dtype=float)
    validate_flow_scenarios(flows, probabilities, node_count=size)
    return FlowScenarios(flows=flows, probabilities=probabilities)


def validate_flow_scenarios(
    flows: np.ndarray, probabilities: np.ndarray, node_count: int | None = None
) -> None:
    """Validate normalized square flow scenarios and their explicit weights."""
    values = np.asarray(flows, dtype=float)
    weights = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 3
        or values.shape[0] == 0
        or values.shape[1] != values.shape[2]
        or (node_count is not None and values.shape[1:] != (node_count, node_count))
    ):
        raise ValueError("scenario flows must have shape (scenarios, nodes, nodes)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scenario flows must contain finite, nonnegative values")
    if not np.all(np.diagonal(values, axis1=1, axis2=2) == 0):
        raise ValueError("scenario flows must have a zero diagonal")
    if not np.allclose(values.sum(axis=(1, 2)), 1.0):
        raise ValueError("each scenario flow matrix must sum to 1")
    if weights.shape != (values.shape[0],):
        raise ValueError("probabilities must match the scenario count")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")
