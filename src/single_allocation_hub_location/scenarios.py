"""Reproducible stochastic flow scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    ``default_rng``. If a Poisson draw has zero total flow, the normalized base
    flow is used for that scenario so every returned scenario remains valid.
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
    normalized_base = base / total
    flows = np.empty((scenario_count, size, size), dtype=float)
    for index in range(scenario_count):
        multipliers = rng.uniform(0.5, 1.5, size=size)
        sampled = rng.poisson(base * multipliers[:, None] * multipliers[None, :])
        sampled_total = sampled.sum()
        flows[index] = normalized_base if sampled_total == 0 else sampled / sampled_total

    probabilities = np.full(scenario_count, 1.0 / scenario_count, dtype=float)
    return FlowScenarios(flows=flows, probabilities=probabilities)
