"""Risk measures for finite scenario distributions."""

from __future__ import annotations

import numpy as np


def conditional_beta_mean(
    costs: np.ndarray, probabilities: np.ndarray, beta: float
) -> float:
    """Return the upper-tail conditional beta-mean (discrete CVaR).

    The calculation is ``min_eta eta + E[(cost - eta)+] / (1 - beta)``.
    This handles a probability atom at the beta quantile fractionally.
    """
    values = np.asarray(costs, dtype=float)
    weights = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("costs and probabilities must be nonempty matching vectors")
    if not np.isfinite(values).all():
        raise ValueError("costs must be finite")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")
    if not 0 <= beta < 1:
        raise ValueError("beta must satisfy 0 <= beta < 1")

    order = np.argsort(values, kind="stable")
    ordered_costs = values[order]
    cumulative = np.cumsum(weights[order])
    quantile_index = min(int(np.searchsorted(cumulative, beta, side="left")), values.size - 1)
    eta = ordered_costs[quantile_index]
    return float(eta + np.dot(weights, np.maximum(values - eta, 0.0)) / (1.0 - beta))
