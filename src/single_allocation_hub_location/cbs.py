"""Compact clustering-based potential hub sets (CBS) heuristic.

This follows Paper B Appendix A.1 (Algorithm 3): rank nodes, retain 2p
potential hubs, build distance clusters, then search feasible hub sets. DL-CBS
changes only the ranking order supplied by DLHr.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from .evaluation import evaluate_risk_objective
from .gvns import HubCandidate, nearest_assignments, validate_candidate


def cbs_node_order(demand: np.ndarray, distance: np.ndarray, scores: np.ndarray | None = None) -> tuple[int, ...]:
    n = demand.shape[0]
    if scores is not None:
        return tuple(sorted(range(n), key=lambda i: (-float(scores[i]), i)))
    out = demand.sum(axis=1); inn = demand.sum(axis=0)
    importance = (out + inn) * np.maximum(distance.sum(axis=1), 1e-12)
    return tuple(sorted(range(n), key=lambda i: (-float(importance[i]), i)))


def create_cbs_clusters(distance: np.ndarray, order: tuple[int, ...], p: int, *, p_c: float = 1.0) -> dict[str, object]:
    n = distance.shape[0]; potential = tuple(order[: min(n, 2*p)])
    if len(potential) <= 1: radius = 0.0
    else: radius = float(sum(np.min(distance[i, [j for j in potential if j != i]]) for i in potential) / (2*p))
    clusters = {i: tuple(j for j in range(n) if j != i and distance[i,j] <= radius) for i in potential}
    return {"potential_hubs": potential, "radius": radius, "clusters": clusters, "p_c": p_c}


def solve_cbs(distance, demand, scenario_flows, probabilities, p, alpha, beta, *, scores=None, max_evaluations=100, time_limit=None, seed=0, method="cbs"):
    d = np.asarray(distance, float); w = np.asarray(demand, float)
    order = cbs_node_order(w, d, scores); info = create_cbs_clusters(d, order, p)
    pool = tuple(info["potential_hubs"])
    if len(pool) < p: pool = tuple(order)
    start=time.perf_counter(); best=None; best_obj=np.inf; evaluations=0
    for hubs in combinations(pool, p):
        if evaluations >= max_evaluations or (time_limit and time.perf_counter()-start >= time_limit): break
        cand=HubCandidate(tuple(sorted(hubs)), nearest_assignments(d, tuple(sorted(hubs))))
        validate_candidate(cand, d.shape[0], p)
        obj=evaluate_risk_objective(cand.assignments,d,scenario_flows,probabilities,alpha,beta); evaluations += 1
        if obj < best_obj: best,best_obj=cand,obj
    if best is None: raise RuntimeError("CBS budget exhausted before a feasible candidate")
    return {"method": method, "candidate": best, "objective": float(best_obj), "runtime": time.perf_counter()-start,
            "evaluation_count": evaluations, "status": "completed" if evaluations < max_evaluations else "budget_limited",
            "proven_optimal": False, "ranker_identity": "scores" if scores is not None else None,
            "potential_hubs": list(pool), "clusters": {str(k): list(v) for k,v in info["clusters"].items()}}


def solve_dl_cbs(*args, scores, **kwargs):
    """DL-CBS: identical CBS search with only DLHr ordering changed."""
    kwargs["method"] = "dl_cbs"
    return solve_cbs(*args, scores=scores, **kwargs)
