"""Restricted CBS and DL-RCBS for the single-allocation project.

Paper B Appendix A.1 (pp. 20--21), Algorithm 3 and Table A.6 define the
potential-set/clustering reduction.  This compact implementation is the RCBS
variant: every non-isolated cluster supplies a hub, isolated centres are hubs,
and hubs are restricted to the constructed candidate region.  DL-RCBS changes
only the importance ordering to the supplied DLHr scores.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from .evaluation import evaluate_risk_objective
from .gvns import HubCandidate, nearest_assignments, validate_candidate


def cbs_node_order(demand: np.ndarray, distance: np.ndarray, scores: np.ndarray | None = None) -> tuple[int, ...]:
    """Paper B's Imp1 ordering, or DLHr ordering for the guided counterpart."""
    demand = np.asarray(demand, float); distance = np.asarray(distance, float)
    n = demand.shape[0]
    if scores is not None:
        values = np.asarray(scores, float)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("scores must be a finite value for every node")
        return tuple(sorted(range(n), key=lambda i: (-float(values[i]), i)))
    produced = demand.sum(axis=1); attracted = demand.sum(axis=0)
    total_distance = distance.sum(axis=1)  # C_v in Paper B Appendix A.1.
    importance = (produced + attracted) * total_distance
    return tuple(sorted(range(n), key=lambda i: (-float(importance[i]), i)))


def create_cbs_clusters(distance: np.ndarray, order: tuple[int, ...], p: int) -> dict[str, object]:
    """Implement Paper B Algorithm 3's potential set and radius construction."""
    d = np.asarray(distance, float); n = d.shape[0]
    if d.shape != (n, n) or not 1 <= p <= n:
        raise ValueError("invalid CBS distance matrix or p")
    potential = tuple(order[:min(2 * p, n)])
    minima = [min(float(d[i, j]) for j in potential if j != i) for i in potential] if len(potential) > 1 else [0.0]
    radius = float(sum(minima) / (2 * p))
    centres: list[int] = []; isolated: list[int] = []; clusters: dict[int, tuple[int, ...]] = {}
    for node in potential:
        if any(node in cluster for cluster in clusters.values()):
            continue
        near_potential = tuple(other for other in potential if other != node and d[node, other] < radius)
        if not near_potential:
            isolated.append(node)
        else:
            centres.append(node)
            clusters[node] = tuple(k for k in range(n) if d[node, k] < radius)
    expanded_isolated = tuple(sorted({k for i in isolated for k in range(n) if d[i, k] <= radius} | set(isolated)))
    covered = set(expanded_isolated)
    for cluster in clusters.values(): covered.update(cluster)
    residual: list[int] = []
    for node in order:
        if node not in covered:
            residual.append(node); covered.add(node)
        if len(centres) + len(isolated) + len(residual) >= p:
            break
    allowed = tuple(sorted(covered))
    return {"potential_hubs": potential, "radius": radius, "centres": tuple(centres),
            "isolated": tuple(isolated), "expanded_isolated": expanded_isolated,
            "clusters": clusters, "residual": tuple(residual), "allowed_hubs": allowed}


def solve_cbs(distance, demand, scenario_flows, probabilities, p, alpha, beta, *, scores=None, max_evaluations=100, time_limit=None, seed=0, method="cbs"):
    """Search the deterministic RCBS-restricted feasible hub combinations."""
    d = np.asarray(distance, float); demand = np.asarray(demand, float)
    order = cbs_node_order(demand, d, scores)
    info = create_cbs_clusters(d, order, p)
    allowed = tuple(info["allowed_hubs"])
    required = set(info["isolated"])
    if len(required) > p:
        required = set(sorted(required, key=lambda i: order.index(i))[:p])
    start = time.perf_counter(); best = None; best_objective = np.inf; evaluations = 0
    for hubs in combinations(allowed, p):
        hub_set = set(hubs)
        if not required.issubset(hub_set):
            continue
        if any(not hub_set.intersection(cluster) for cluster in info["clusters"].values()):
            continue
        if evaluations >= max_evaluations or (time_limit is not None and time.perf_counter() - start >= time_limit):
            break
        candidate = HubCandidate(tuple(sorted(hubs)), nearest_assignments(d, tuple(sorted(hubs))))
        validate_candidate(candidate, d.shape[0], p)
        objective = evaluate_risk_objective(candidate.assignments, d, scenario_flows, probabilities, alpha, beta)
        evaluations += 1
        if objective < best_objective - 1e-12:
            best, best_objective = candidate, float(objective)
    if best is None:
        # The paper's restrictions can be over-constraining on a small adapted
        # data set; fall back to the potential set only, never an infeasible state.
        for hubs in combinations(info["potential_hubs"], p):
            candidate = HubCandidate(tuple(sorted(hubs)), nearest_assignments(d, tuple(sorted(hubs))))
            validate_candidate(candidate, d.shape[0], p)
            objective = evaluate_risk_objective(candidate.assignments, d, scenario_flows, probabilities, alpha, beta)
            evaluations += 1
            if objective < best_objective:
                best, best_objective = candidate, float(objective)
            if evaluations >= max_evaluations: break
    if best is None:
        raise RuntimeError("CBS budget exhausted before a feasible candidate")
    return {"method": method, "candidate": best, "objective": best_objective,
            "runtime": time.perf_counter() - start, "evaluation_count": evaluations,
            "status": "completed" if evaluations < max_evaluations else "budget_limited",
            "proven_optimal": False, "ranker_identity": "provided_scores" if scores is not None else None,
            "potential_hubs": list(info["potential_hubs"]), "clusters": {str(k): list(v) for k, v in info["clusters"].items()},
            "cbs_variant": "RCBS", "search_path": "cbs_rcbs_clustered_hub_combination_enumeration"}


def solve_dl_cbs(*args, scores, **kwargs):
    """DL-RCBS: identical constraints/search with only DLHr ranking substituted."""
    kwargs["method"] = "dl_cbs"
    return solve_cbs(*args, scores=scores, **kwargs)
