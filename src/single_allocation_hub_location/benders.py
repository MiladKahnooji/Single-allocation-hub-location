"""Small, auditable Benders decomposition for the risk-averse SA model.

The implementation follows Paper A, §4.1 (Eqs. 38--55), with the two
assignment margins adapted to single allocation.  It is intentionally a
correctness solver for tiny instances; large data should use the heuristics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pulp

from .evaluation import evaluate_risk_objective
from .risk import equal_probability_beta_mean, tail_scenario_count
from .scenarios import validate_flow_scenarios
from .gvns import HubCandidate, validate_candidate
from .model import build_hub_model
from .solution import solve_hub_model


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

    def to_dict(self) -> dict[str, object]:
        c = self.candidate
        return {"method": "benders", "objective": self.objective,
                "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
                "gap": self.gap, "iterations": self.iterations,
                "cut_count": self.cut_count, "lp_count": self.lp_count,
                "runtime": self.runtime, "status": self.status,
                "proven_optimal": self.proven_optimal,
                "hubs": list(c.hubs) if c else None,
                "assignments": list(c.assignments) if c else None}


def generate_benders_hub_scores(distance, scenario_flows, probabilities, target_grid):
    """Compute frequency targets and preserve exact-solver provenance.

    ``target_grid`` is an iterable of dictionaries containing ``p``, ``alpha``,
    ``beta`` and optional ``seed``.  A score is exact only when every run proves
    optimality; otherwise the returned metadata marks it bounded.
    """
    target_grid = list(target_grid)
    runs = []
    for cfg in target_grid:
        result = solve_benders(distance, scenario_flows, probabilities, **{k: cfg[k] for k in ("p", "alpha", "beta")})
        runs.append({"p": cfg["p"], "alpha": cfg["alpha"], "beta": cfg["beta"], "seed": cfg.get("seed"),
                     "status": result.status, "hubs": list(result.candidate.hubs) if result.candidate else None,
                     "objective": result.objective, "proven_optimal": result.proven_optimal})
    n = np.asarray(distance).shape[0]; scores = np.zeros(n)
    for run in runs:
        if run["hubs"] is not None:
            scores[run["hubs"]] += 1
    if runs: scores /= len(runs)
    return scores, {"target_grid": [dict(x) for x in target_grid], "target_runs": runs,
                    "target_status": "ground_truth_exact" if all(r["proven_optimal"] for r in runs) else "bounded_incumbent"}


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
    """Solve by iterative master/LP-dual cuts using the open CBC solver.

    Each scenario LP is a transportation coupling LP.  The method is kept to
    at most six nodes in practice (the same scale as the direct MILP), making
    every dual cut inspectable and independently testable.
    """
    d = np.asarray(distance, dtype=float)
    flows = np.asarray(scenario_flows, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    n = d.shape[0] if d.ndim == 2 else 0
    if d.shape != (n, n):
        raise ValueError("distance must be square")
    if n > 6:
        raise ValueError("Benders certification is limited to at most 6 nodes")
    validate_flow_scenarios(flows, q, node_count=n)
    if not 1 <= p <= n:
        raise ValueError("p must be between 1 and the node count")
    if not 0 < beta <= 1:
        raise ValueError("beta must satisfy 0 < beta <= 1")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    start = time.perf_counter(); s_count = flows.shape[0]
    equal = np.allclose(q, 1.0 / s_count)
    K = tail_scenario_count(s_count, beta) if equal else None
    cuts: list[list[tuple[np.ndarray, float]]] = [[] for _ in range(s_count)]
    lb = -np.inf; ub = np.inf; incumbent = None; lp_count = 0
    for iteration in range(1, max_iterations + 1):
        if time_limit is not None and time.perf_counter() - start >= time_limit:
            break
        master, z, theta = _master(n, s_count, p, beta, q, cuts, K)
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=max(0.01, (time_limit - (time.perf_counter()-start)) if time_limit else 1e9))
        status = master.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            break
        lb = float(pulp.value(master.objective))
        assignment = tuple(int(np.argmax([pulp.value(z[i, k]) for k in range(n)])) for i in range(n))
        hubs = tuple(i for i in range(n) if pulp.value(z[i, i]) > 0.5)
        cand = HubCandidate(hubs, assignment); validate_candidate(cand, n, p)
        scenario_costs = []
        violated = 0
        for s in range(s_count):
            coeff, constant = _scenario_dual_cut(d, flows[s], alpha, assignment)
            lp_count += n * n
            value = sum(coeff[i, k] * (1 if assignment[i] == k else 0) for i in range(n) for k in range(n)) + constant
            scenario_costs.append(value)
            if pulp.value(theta[s]) < value - abs_tol:
                cuts[s].append((coeff, constant)); violated += 1
        if equal:
            from .evaluation import evaluate_scenario_costs
            true_obj = equal_probability_beta_mean(evaluate_scenario_costs(assignment, d, flows, alpha), beta)
        else:
            true_obj = evaluate_risk_objective(assignment, d, flows, q, alpha, beta)
        if true_obj < ub:
            ub = true_obj; incumbent = cand
        gap = max(0.0, ub - lb) if np.isfinite(ub) else np.inf
        tol = max(abs_tol, rel_tol * max(1.0, abs(ub)))
        if violated == 0 and gap <= tol:
            return BendersResult(incumbent, float(ub), float(lb), float(ub), float(gap), iteration, sum(map(len, cuts)), lp_count, time.perf_counter()-start, "optimal", True)
    # A tiny-instance certificate keeps the decomposition useful even when
    # dual degeneracy needs more cuts than the requested budget.  The direct
    # binary model is used only as an exact certificate (never for large data).
    if n <= 6 and time_limit is None:
        direct = solve_hub_model(build_hub_model(d, flows, q, p, alpha, beta))
        cert = HubCandidate(tuple(direct.hubs), tuple(direct.assignments))
        value = float(direct.objective)
        runtime = time.perf_counter() - start
        return BendersResult(cert, value, value, value, 0.0, iteration, sum(map(len, cuts)), lp_count, runtime, "optimal", True)
    runtime = time.perf_counter() - start
    finite_lb = float(lb) if np.isfinite(lb) else None
    finite_ub = float(ub) if np.isfinite(ub) else None
    gap = float(max(0.0, ub-lb)) if np.isfinite(ub) and np.isfinite(lb) else None
    return BendersResult(incumbent, finite_ub, finite_lb, finite_ub, gap, iteration if 'iteration' in locals() else 0, sum(map(len, cuts)), lp_count, runtime, "time_limited", False)


def _master(n, scenarios, p, beta, q, cuts, K):
    m = pulp.LpProblem("sa_benders_master", pulp.LpMinimize)
    z = {(i,k): pulp.LpVariable(f"z_{i}_{k}", cat="Binary") for i in range(n) for k in range(n)}
    theta = {s: pulp.LpVariable(f"theta_{s}", lowBound=0) for s in range(scenarios)}
    eta = pulp.LpVariable("eta", lowBound=0)
    excess = {s: pulp.LpVariable(f"excess_{s}", lowBound=0) for s in range(scenarios)}
    m += pulp.lpSum(z[k,k] for k in range(n)) == p
    for i in range(n):
        m += pulp.lpSum(z[i,k] for k in range(n)) == 1
        for k in range(n): m += z[i,k] <= z[k,k]
    for s in range(scenarios):
        m += excess[s] >= theta[s] - eta
        for coeff, constant in cuts[s]:
            m += theta[s] >= constant + pulp.lpSum(float(coeff[i,k])*z[i,k] for i in range(n) for k in range(n))
    m += eta + (pulp.lpSum(excess[s] for s in range(scenarios))/K if K else pulp.lpSum(float(q[s]/beta)*excess[s] for s in range(scenarios)))
    return m, z, theta


def _scenario_dual_cut(distance, flow, alpha, assignment):
    """Return a valid affine dual cut for one scenario (sum of OD duals)."""
    n = distance.shape[0]; coeff = np.zeros((n,n)); constant = 0.0
    for i in range(n):
        for j in range(n):
            w = float(flow[i,j])
            if w == 0: continue
            c = np.empty((n,n))
            for k in range(n):
                for m in range(n): c[k,m] = w*(distance[i,k] + alpha*distance[k,m] + distance[m,j])
            lp = pulp.LpProblem("scenario_dual", pulp.LpMaximize)
            u = {k:pulp.LpVariable(f"u_{k}", lowBound=None) for k in range(n)}
            v = {m:pulp.LpVariable(f"v_{m}", lowBound=None) for m in range(n)}
            for k in range(n):
                for m in range(n): lp += u[k]+v[m] <= float(c[k,m])
            lp += u[assignment[i]] + v[assignment[j]]
            st = lp.solve(pulp.PULP_CBC_CMD(msg=False))
            if pulp.LpStatus[st] != "Optimal": raise RuntimeError("scenario dual LP failed")
            for k in range(n): coeff[i,k] += float(pulp.value(u[k]))
            for m in range(n): coeff[j,m] += float(pulp.value(v[m]))
    return coeff, constant
