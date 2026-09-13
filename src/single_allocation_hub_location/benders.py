"""Iterative Benders cuts for risk-averse single-allocation hub location.

Paper A, §4.1 (pp. 7--9) separates route decisions and derives dual cuts by
inspection.  This module adapts that idea to the project’s binary single
allocation variables: each fixed-assignment scenario subproblem is a balanced
transportation LP whose dual gives a globally valid cost cut.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pulp

from .evaluation import evaluate_scenario_costs
from .gvns import HubCandidate, nearest_assignments, validate_candidate
from .risk import conditional_beta_mean, equal_probability_beta_mean, tail_scenario_count
from .scenarios import validate_flow_scenarios


@dataclass(frozen=True)
class BendersResult:
    candidate: HubCandidate | None
    objective: float | None
    lower_bound: float | None
    upper_bound: float | None
    gap: float | None
    iterations: int
    cut_count: int
    lp_count: int
    runtime: float
    status: str
    proven_optimal: bool
    scenario_costs: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        candidate = self.candidate
        return {
            "method": "benders", "objective": self.objective,
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "gap": self.gap, "iterations": self.iterations,
            "cut_count": self.cut_count, "lp_count": self.lp_count,
            "runtime": self.runtime, "status": self.status,
            "proven_optimal": self.proven_optimal,
            "hubs": list(candidate.hubs) if candidate else None,
            "assignments": list(candidate.assignments) if candidate else None,
            "scenario_costs": list(self.scenario_costs),
        }


def solve_benders(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    p: int,
    alpha: float,
    beta: float,
    *,
    max_iterations: int = 50,
    time_limit: float | None = None,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> BendersResult:
    """Run a deterministic Benders loop with certified bounds when possible.

    A stopped run always returns a feasible incumbent (unless master creation
    itself fails), its best available lower bound, and ``proven_optimal=False``.
    The implementation accepts CAB/AP dimensions; users should bound those
    runs because every iteration creates one cut per scenario.
    """
    d, flows, q = _validate(distance, scenario_flows, probabilities, p, alpha, beta)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive when supplied")
    n, scenario_count = d.shape[0], flows.shape[0]
    equal = np.allclose(q, 1.0 / scenario_count)
    tail_count = tail_scenario_count(scenario_count, beta) if equal else None
    started = time.perf_counter()
    cuts: list[list[np.ndarray]] = [[] for _ in range(scenario_count)]
    incumbent = _initial_candidate(d, p)
    incumbent_costs = evaluate_scenario_costs(incumbent.assignments, d, flows, alpha)
    upper = _risk(incumbent_costs, q, beta, equal)
    lower: float | None = None
    lp_count = 0
    iterations = 0
    termination = "iteration_limit"

    for iterations in range(1, max_iterations + 1):
        remaining = _remaining(started, time_limit)
        if remaining is not None and remaining <= 0:
            termination = "time_limited"
            break
        master, z, theta = _build_master(n, scenario_count, p, beta, q, cuts, tail_count)
        status = master.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=remaining))
        if pulp.LpStatus[status] != "Optimal":
            termination = "time_limited" if time_limit is not None else "master_not_optimal"
            break
        lower = float(pulp.value(master.objective))
        assignment = tuple(
            int(np.argmax([float(pulp.value(z[i, k]) or 0.0) for k in range(n)]))
            for i in range(n)
        )
        hubs = tuple(k for k in range(n) if float(pulp.value(z[k, k]) or 0.0) > 0.5)
        candidate = HubCandidate(hubs, assignment)
        validate_candidate(candidate, n, p)
        scenario_costs = np.empty(scenario_count, dtype=float)
        violated = 0
        interrupted = False
        for scenario in range(scenario_count):
            if (_remaining(started, time_limit) is not None
                    and _remaining(started, time_limit) <= 0):
                interrupted = True
                termination = "time_limited"
                break
            cut, cost = scenario_dual_cut(d, flows[scenario], alpha, assignment)
            lp_count += 1
            scenario_costs[scenario] = cost
            theta_value = float(pulp.value(theta[scenario]) or 0.0)
            if theta_value < cost - abs_tol:
                cuts[scenario].append(cut)
                violated += 1
        if interrupted:
            break
        value = _risk(scenario_costs, q, beta, equal)
        if value < upper - abs_tol:
            incumbent, incumbent_costs, upper = candidate, scenario_costs, value
        gap = max(0.0, upper - lower)
        tolerance = max(abs_tol, rel_tol * max(1.0, abs(upper)))
        if violated == 0 and gap <= tolerance:
            return BendersResult(
                incumbent, float(upper), float(lower), float(upper), float(gap),
                iterations, sum(map(len, cuts)), lp_count, time.perf_counter() - started,
                "optimal", True, tuple(float(x) for x in incumbent_costs),
            )
    gap = None if lower is None else float(max(0.0, upper - lower))
    return BendersResult(
        incumbent, float(upper), lower, float(upper), gap, iterations,
        sum(map(len, cuts)), lp_count, time.perf_counter() - started,
        termination, False, tuple(float(x) for x in incumbent_costs),
    )


def scenario_dual_cut(
    distance: np.ndarray, flow: np.ndarray, alpha: float, assignment: tuple[int, ...]
) -> tuple[np.ndarray, float]:
    """Solve a scenario transportation dual by inspection.

    For OD ``(i,j)``, choose ``u_k=C[k,m0]`` and
    ``v_m=min_k(C[k,m]-u_k)``, where ``m0`` is j's assigned hub.  Thus
    ``u_k+v_m<=C[k,m]`` for all k,m and, at the incumbent, its value is exactly
    ``C[k0,m0]``. Summing OD duals yields a valid Benders optimality cut.
    """
    d = np.asarray(distance, dtype=float); w = np.asarray(flow, dtype=float)
    n = d.shape[0]
    assigned = np.asarray(assignment, dtype=int)
    if d.shape != (n, n) or w.shape != (n, n) or assigned.shape != (n,):
        raise ValueError("scenario dual dimensions must match")
    coefficient = np.zeros((n, n), dtype=float)
    objective = 0.0
    inter = alpha * d
    for i in range(n):
        origin = d[i, :]
        k0 = assigned[i]
        for j in range(n):
            amount = float(w[i, j])
            if amount == 0.0:
                continue
            m0 = assigned[j]
            cost = amount * (origin[:, None] + inter + d[:, j][None, :])
            u = cost[:, m0]
            v = np.min(cost - u[:, None], axis=0)
            coefficient[i, :] += u
            coefficient[j, :] += v
            objective += float(u[k0] + v[m0])
    return coefficient, objective


def generate_benders_hub_scores(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    target_grid: Iterable[dict[str, object]],
    *,
    max_iterations: int = 100,
    time_limit: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate frequency targets with full exact/bounded provenance."""
    grid = tuple(dict(item) for item in target_grid)
    scores = np.zeros(np.asarray(distance).shape[0], dtype=float)
    runs: list[dict[str, object]] = []
    for item in grid:
        result = solve_benders(
            distance, scenario_flows, probabilities,
            int(item["p"]), float(item["alpha"]), float(item["beta"]),
            max_iterations=max_iterations, time_limit=time_limit,
        )
        if result.candidate is not None:
            scores[list(result.candidate.hubs)] += 1.0
        runs.append({
            "p": item["p"], "alpha": item["alpha"], "beta": item["beta"],
            "seed": item.get("seed"), "status": result.status,
            "hubs": tuple(result.candidate.hubs) if result.candidate else (),
            "objective": result.objective, "proven_optimal": result.proven_optimal,
            "lower_bound": result.lower_bound, "upper_bound": result.upper_bound,
            "gap": result.gap,
        })
    if grid:
        scores /= len(grid)
    exact = bool(runs) and all(bool(run["proven_optimal"]) for run in runs)
    return scores, {
        "target_grid": grid, "target_runs": tuple(runs),
        "hub_scores_status": "ground_truth_exact" if exact else "bounded_incumbent",
        "all_target_solutions_proven_optimal": exact,
    }


def _build_master(n, scenarios, p, beta, probabilities, cuts, tail_count):
    model = pulp.LpProblem("single_allocation_benders_master", pulp.LpMinimize)
    z = {(i, k): pulp.LpVariable(f"z_{i}_{k}", cat="Binary") for i in range(n) for k in range(n)}
    theta = {s: pulp.LpVariable(f"theta_{s}", lowBound=0) for s in range(scenarios)}
    eta = pulp.LpVariable("eta", lowBound=0)
    excess = {s: pulp.LpVariable(f"excess_{s}", lowBound=0) for s in range(scenarios)}
    model += pulp.lpSum(z[k, k] for k in range(n)) == p
    for i in range(n):
        model += pulp.lpSum(z[i, k] for k in range(n)) == 1
        for k in range(n):
            model += z[i, k] <= z[k, k]
    for s in range(scenarios):
        model += excess[s] >= theta[s] - eta
        for cut in cuts[s]:
            model += theta[s] >= pulp.lpSum(float(cut[i, k]) * z[i, k] for i in range(n) for k in range(n))
    if tail_count is not None:
        model += eta + pulp.lpSum(excess.values()) / tail_count
    else:
        model += eta + pulp.lpSum(float(probabilities[s] / beta) * excess[s] for s in range(scenarios))
    return model, z, theta


def _initial_candidate(distance: np.ndarray, p: int) -> HubCandidate:
    # Deterministic dispersed hubs supply an incumbent before a time-limited master.
    n = distance.shape[0]
    hubs = [0]
    while len(hubs) < p:
        candidates = [node for node in range(n) if node not in hubs]
        hubs.append(max(candidates, key=lambda node: (min(distance[node, h] for h in hubs), -node)))
    ordered = tuple(sorted(hubs))
    candidate = HubCandidate(ordered, nearest_assignments(distance, ordered))
    validate_candidate(candidate, n, p)
    return candidate


def _risk(costs, probabilities, beta, equal):
    if equal:
        return equal_probability_beta_mean(np.asarray(costs), beta)
    return conditional_beta_mean(np.asarray(costs), np.asarray(probabilities), beta)


def _remaining(started: float, time_limit: float | None) -> float | None:
    return None if time_limit is None else max(0.0, time_limit - (time.perf_counter() - started))


def _validate(distance, scenario_flows, probabilities, p, alpha, beta):
    d = np.asarray(distance, dtype=float); flows = np.asarray(scenario_flows, dtype=float); q = np.asarray(probabilities, dtype=float)
    n = d.shape[0] if d.ndim == 2 else 0
    if d.shape != (n, n) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("distance must be a finite nonnegative square matrix")
    validate_flow_scenarios(flows, q, node_count=n)
    if not isinstance(p, (int, np.integer)) or not 1 <= p <= n:
        raise ValueError("p must be between 1 and the node count")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not 0 < beta <= 1:
        raise ValueError("beta must satisfy 0 < beta <= 1")
    return d, flows, q
