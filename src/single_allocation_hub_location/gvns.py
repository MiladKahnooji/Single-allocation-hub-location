"""Compact seeded GVNS and score-guided DL-GVNS for feasible hub candidates."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .evaluation import evaluate_risk_objective
from .scenarios import validate_flow_scenarios
from .training_data import HubFeatureData, build_graphs, build_node_features, classical_mds

Method = Literal["gvns", "dl_gvns"]


@dataclass(frozen=True)
class HubCandidate:
    """A single-allocation candidate with explicit open hubs and assignments."""

    hubs: tuple[int, ...]
    assignments: tuple[int, ...]


@dataclass(frozen=True)
class GVNSConfig:
    """Deterministic evaluation budget and optional wall-clock safety limit."""

    p: int
    alpha: float
    beta: float
    max_iterations: int = 8
    max_evaluations: int = 30
    time_limit: float | None = None
    larger_shake_size: int = 2


@dataclass(frozen=True)
class GVNSResult:
    """One heuristic result with the complete feasible incumbent."""

    method: Method
    candidate: HubCandidate
    objective: float
    runtime: float
    evaluation_count: int
    iteration_count: int
    status: str
    ranker_identity: str | None = None

    @property
    def proven_optimal(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "objective": self.objective,
            "runtime": self.runtime,
            "evaluation_count": self.evaluation_count,
            "iteration_count": self.iteration_count,
            "hubs": self.candidate.hubs,
            "assignments": self.candidate.assignments,
            "status": self.status,
            "proven_optimal": False,
            "ranker_identity": self.ranker_identity,
        }


class CandidateEvaluator:
    """Validate candidates and evaluate the approved route/risk objective."""

    def __init__(
        self,
        distance: np.ndarray,
        scenario_flows: np.ndarray,
        probabilities: np.ndarray,
        config: GVNSConfig,
    ) -> None:
        self.distance = np.asarray(distance, dtype=float)
        self.flows = np.asarray(scenario_flows, dtype=float)
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.config = config
        self._validate_inputs()
        self.evaluation_count = 0
        self._cache: dict[HubCandidate, float] = {}

    @property
    def node_count(self) -> int:
        return self.distance.shape[0]

    def evaluate(self, candidate: HubCandidate) -> float:
        """Validate before every non-cached objective evaluation."""
        validate_candidate(candidate, self.node_count, self.config.p)
        cached = self._cache.get(candidate)
        if cached is not None:
            return cached
        if self.evaluation_count >= self.config.max_evaluations:
            raise RuntimeError("GVNS evaluation budget exhausted")
        value = evaluate_risk_objective(
            candidate.assignments,
            self.distance,
            self.flows,
            self.probabilities,
            self.config.alpha,
            self.config.beta,
        )
        if not np.isfinite(value):
            raise RuntimeError("candidate objective must be finite")
        self.evaluation_count += 1
        self._cache[candidate] = value
        return value

    def _validate_inputs(self) -> None:
        size = self.distance.shape[0]
        if self.distance.shape != (size, size):
            raise ValueError("distance must be a square matrix")
        if not np.isfinite(self.distance).all() or (self.distance < 0).any():
            raise ValueError("distance must contain finite, nonnegative values")
        if not 1 <= self.config.p <= size:
            raise ValueError("p must be between 1 and the node count")
        if not np.isfinite(self.config.alpha) or self.config.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not 0 < self.config.beta <= 1:
            raise ValueError("beta must satisfy 0 < beta <= 1")
        if self.config.max_iterations <= 0 or self.config.max_evaluations <= 0:
            raise ValueError("iteration and evaluation limits must be positive")
        if self.config.time_limit is not None and self.config.time_limit <= 0:
            raise ValueError("time_limit must be positive when supplied")
        validate_flow_scenarios(self.flows, self.probabilities, node_count=size)


def validate_candidate(candidate: HubCandidate, node_count: int, p: int) -> None:
    """Enforce exactly p distinct hubs and a complete feasible allocation."""
    if len(candidate.hubs) != p or len(set(candidate.hubs)) != p:
        raise ValueError("candidate must contain exactly p distinct hubs")
    hubs = set(candidate.hubs)
    if any(not 0 <= hub < node_count for hub in hubs):
        raise ValueError("candidate hubs must be valid node indices")
    if len(candidate.assignments) != node_count:
        raise ValueError("candidate must assign every node")
    if any(assigned not in hubs for assigned in candidate.assignments):
        raise ValueError("candidate assignments must use open hubs")
    if any(candidate.assignments[hub] != hub for hub in hubs):
        raise ValueError("every open hub must be assigned to itself")


def nearest_assignments(distance: np.ndarray, hubs: tuple[int, ...]) -> tuple[int, ...]:
    """Assign nodes to the nearest open hub with deterministic hub-index ties."""
    hub_array = np.asarray(sorted(hubs), dtype=int)
    distances = np.asarray(distance, dtype=float)
    proximity = distances[:, hub_array] + distances[hub_array, :].T
    assigned = hub_array[np.argmin(proximity, axis=1)]
    assigned[hub_array] = hub_array
    return tuple(int(value) for value in assigned)


def initialize_candidate(
    distance: np.ndarray, p: int, rng: np.random.Generator, scores: np.ndarray | None = None
) -> HubCandidate:
    """Create either an independent random or score-prioritized feasible start."""
    size = np.asarray(distance).shape[0]
    if scores is None:
        hubs = tuple(sorted(int(value) for value in rng.choice(size, size=p, replace=False)))
    else:
        priorities = _score_order(scores, descending=True)
        hubs = tuple(sorted(int(value) for value in priorities[:p]))
    candidate = HubCandidate(hubs, nearest_assignments(distance, hubs))
    validate_candidate(candidate, size, p)
    return candidate


def shake_candidate(
    candidate: HubCandidate,
    distance: np.ndarray,
    p: int,
    rng: np.random.Generator,
    swap_count: int,
    scores: np.ndarray | None = None,
) -> HubCandidate:
    """Shake with one or more hub swaps, always rebuilding feasible assignments."""
    size = np.asarray(distance).shape[0]
    validate_candidate(candidate, size, p)
    if swap_count <= 0 or p == size:
        return candidate
    hubs = set(candidate.hubs)
    # Use each original hub and closed node at most once.  A multi-swap is
    # therefore genuinely larger than a single swap whenever enough nodes are
    # available, rather than immediately undoing a prior replacement.
    removable = set(hubs)
    available_incoming = set(range(size)) - hubs
    for _ in range(min(swap_count, p, size - p)):
        if scores is None:
            outgoing = int(rng.choice(tuple(sorted(removable))))
            incoming = int(rng.choice(tuple(sorted(available_incoming))))
        else:
            outgoing = min(removable, key=lambda node: (float(scores[node]), node))
            incoming = min(
                available_incoming, key=lambda node: (-float(scores[node]), node)
            )
        hubs.remove(outgoing)
        hubs.add(incoming)
        removable.remove(outgoing)
        available_incoming.remove(incoming)
    ordered = tuple(sorted(hubs))
    shaken = HubCandidate(ordered, nearest_assignments(distance, ordered))
    validate_candidate(shaken, size, p)
    return shaken


def node_reassignment_neighbors(candidate: HubCandidate) -> tuple[HubCandidate, ...]:
    """Return all feasible one-node reassignment neighbors in stable order."""
    hubs = tuple(sorted(candidate.hubs))
    neighbors: list[HubCandidate] = []
    for node, current in enumerate(candidate.assignments):
        if node in hubs:
            continue
        for replacement in hubs:
            if replacement == current:
                continue
            assignments = list(candidate.assignments)
            assignments[node] = replacement
            neighbors.append(HubCandidate(hubs, tuple(assignments)))
    return tuple(neighbors)


def hub_swap_neighbors(
    candidate: HubCandidate, distance: np.ndarray, scores: np.ndarray | None = None
) -> tuple[HubCandidate, ...]:
    """Return all feasible one-hub swaps, optionally score-prioritized."""
    size = np.asarray(distance).shape[0]
    hubs = tuple(sorted(candidate.hubs))
    closed = tuple(node for node in range(size) if node not in hubs)
    if scores is not None:
        hubs = tuple(sorted(hubs, key=lambda node: (float(scores[node]), node)))
        closed = tuple(sorted(closed, key=lambda node: (-float(scores[node]), node)))
    neighbors: list[HubCandidate] = []
    for outgoing in hubs:
        for incoming in closed:
            new_hubs = tuple(sorted((set(hubs) - {outgoing}) | {incoming}))
            neighbors.append(HubCandidate(new_hubs, nearest_assignments(distance, new_hubs)))
    return tuple(neighbors)


def solve_gvns(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    config: GVNSConfig,
    search_seed: int,
) -> GVNSResult:
    """Run baseline GVNS from an independent random initialization."""
    return _solve(
        "gvns", distance, scenario_flows, probabilities, config, search_seed, None, None
    )


def solve_dl_gvns(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    config: GVNSConfig,
    search_seed: int,
    rank_scores: np.ndarray,
    ranker_identity: str = "provided_scores",
) -> GVNSResult:
    """Run GVNS guided only by fixed DLHr ranking scores."""
    scores = np.asarray(rank_scores, dtype=float)
    if scores.shape != (np.asarray(distance).shape[0],) or not np.isfinite(scores).all():
        raise ValueError("rank_scores must be one finite value per node")
    return _solve(
        "dl_gvns",
        distance,
        scenario_flows,
        probabilities,
        config,
        search_seed,
        scores,
        ranker_identity,
    )


def load_rank_scores(
    distance: np.ndarray, demand: np.ndarray, checkpoint_path: str | Path
) -> tuple[np.ndarray, str]:
    """Load a CPU ranker checkpoint and score features without CAB/AP labels."""
    from .ranker import load_ranker_checkpoint, rank_hubs

    # The supplied cost matrices can retain irrelevant self-costs.  Feature
    # data models geometric distances and validates a zero diagonal, so make a
    # private inference-only copy.  The search evaluator still receives the
    # untouched project distance matrix.
    distances = np.asarray(distance, dtype=float).copy()
    np.fill_diagonal(distances, 0.0)
    demands = np.asarray(demand, dtype=float).copy()
    np.fill_diagonal(demands, 0.0)
    coordinates = classical_mds(distances)
    graphs = build_graphs(demands, distances)
    features = build_node_features(coordinates, distances, demands)
    size = distances.shape[0]
    inference_data = HubFeatureData(
        coordinates=coordinates,
        distance=distances,
        demand=demands,
        production_graph=graphs[0],
        attraction_graph=graphs[1],
        spatial_graph=graphs[2],
        features=features,
        hub_labels=np.zeros(size, dtype=np.int8),
        hub_scores=np.zeros(size, dtype=float),
        metadata={},
    )
    path = Path(checkpoint_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    ranking = rank_hubs(load_ranker_checkpoint(path), inference_data)
    return ranking.scores, f"checkpoint:{path.name}:{digest}"


def _solve(
    method: Method,
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    config: GVNSConfig,
    search_seed: int,
    scores: np.ndarray | None,
    ranker_identity: str | None,
) -> GVNSResult:
    started = time.perf_counter()
    evaluator = CandidateEvaluator(distance, scenario_flows, probabilities, config)
    rng = np.random.default_rng(search_seed)
    incumbent = initialize_candidate(distance, config.p, rng, scores)
    incumbent_value = evaluator.evaluate(incumbent)
    iteration = 0
    status = "evaluation budget reached"
    while iteration < config.max_iterations and evaluator.evaluation_count < config.max_evaluations:
        if config.time_limit is not None and time.perf_counter() - started >= config.time_limit:
            status = "time limit reached"
            break
        depth = 1 if iteration % 2 == 0 else config.larger_shake_size
        shaken = shake_candidate(incumbent, distance, config.p, rng, depth, scores)
        try:
            candidate, candidate_value = _local_search(shaken, evaluator, scores)
        except RuntimeError:
            break
        if candidate_value < incumbent_value - 1e-12:
            incumbent, incumbent_value = candidate, candidate_value
        iteration += 1
    else:
        status = "iteration limit reached"
    if evaluator.evaluation_count >= config.max_evaluations:
        status = "evaluation budget reached"
    return GVNSResult(
        method=method,
        candidate=incumbent,
        objective=incumbent_value,
        runtime=time.perf_counter() - started,
        evaluation_count=evaluator.evaluation_count,
        iteration_count=iteration,
        status=status,
        ranker_identity=ranker_identity,
    )


def _local_search(
    candidate: HubCandidate, evaluator: CandidateEvaluator, scores: np.ndarray | None
) -> tuple[HubCandidate, float]:
    current = candidate
    current_value = evaluator.evaluate(current)
    improved = True
    while improved and evaluator.evaluation_count < evaluator.config.max_evaluations:
        improved = False
        best = current
        best_value = current_value
        neighbors = hub_swap_neighbors(current, evaluator.distance, scores) + node_reassignment_neighbors(current)
        for neighbor in neighbors:
            if evaluator.evaluation_count >= evaluator.config.max_evaluations:
                break
            value = evaluator.evaluate(neighbor)
            if value < best_value - 1e-12:
                best, best_value, improved = neighbor, value, True
        current, current_value = best, best_value
    return current, current_value


def _score_order(scores: np.ndarray, descending: bool) -> np.ndarray:
    indices = np.arange(scores.size)
    return np.lexsort((indices, -scores if descending else scores))
