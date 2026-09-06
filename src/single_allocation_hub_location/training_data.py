"""Lightweight graph and feature inputs for a later hub-ranking model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping

import numpy as np

from .heuristic import solve_hub_heuristic
from .model import build_hub_model
from .solution import HubSolution, solve_hub_model

LabelMethod = Literal["exact", "heuristic"]
_MIN_DISTANCE = 1e-12


@dataclass(frozen=True)
class HubFeatureData:
    """Validated node-level inputs and hub labels for one instance."""

    coordinates: np.ndarray
    distance: np.ndarray
    demand: np.ndarray
    production_graph: np.ndarray
    attraction_graph: np.ndarray
    spatial_graph: np.ndarray
    features: np.ndarray
    hub_labels: np.ndarray
    hub_scores: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_feature_data(self)


def classical_mds(distance: np.ndarray) -> np.ndarray:
    """Derive deterministic two-dimensional coordinates by classical MDS."""
    distances = _validate_square_matrix(distance, "distance", symmetric=True)
    if not np.allclose(np.diag(distances), 0.0):
        raise ValueError("distance must have a zero diagonal")
    distances = (distances + distances.T) / 2.0

    size = distances.shape[0]
    centering = np.eye(size) - np.full((size, size), 1.0 / size)
    gram = -0.5 * centering @ np.square(distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues, kind="stable")[::-1][:2]
    values = np.maximum(eigenvalues[order], 0.0)
    coordinates = eigenvectors[:, order] * np.sqrt(values)
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))

    # Eigenvector signs are arbitrary. Pin each axis to its largest-magnitude
    # coordinate so repeated calls have the same orientation.
    for axis in range(2):
        pivot = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[pivot, axis] < 0:
            coordinates[:, axis] *= -1
    if not np.isfinite(coordinates).all():
        raise ValueError("MDS produced non-finite coordinates")
    return coordinates


def build_graphs(
    demand: np.ndarray, distance: np.ndarray, p_w: float = 0.8, p_c: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build weighted outgoing, incoming, and symmetric nearest-node graphs."""
    demands = _validate_square_matrix(demand, "demand")
    distances = _validate_square_matrix(distance, "distance", symmetric=True)
    if demands.shape != distances.shape:
        raise ValueError("demand and distance shapes must match")
    if not 0 < p_w <= 1:
        raise ValueError("p_w must satisfy 0 < p_w <= 1")
    if not 0 < p_c <= 1:
        raise ValueError("p_c must satisfy 0 < p_c <= 1")

    size = demands.shape[0]
    production = np.zeros((size, size), dtype=float)
    attraction = np.zeros((size, size), dtype=float)
    for node in range(size):
        outgoing = _coverage_neighbors(demands[node], node, p_w)
        incoming = _coverage_neighbors(demands[:, node], node, p_w)
        production[node, outgoing] = demands[node, outgoing]
        attraction[node, incoming] = demands[incoming, node]

    symmetric_distance = (distances + distances.T) / 2.0
    selected = np.zeros((size, size), dtype=bool)
    neighbor_count = min(size - 1, max(1, int(np.ceil(p_c * (size - 1)))))
    indices = np.arange(size)
    for node in range(size):
        candidates = indices[indices != node]
        order = np.lexsort((candidates, symmetric_distance[node, candidates]))
        selected[node, candidates[order[:neighbor_count]]] = True
    selected |= selected.T
    spatial = np.zeros((size, size), dtype=float)
    rows, columns = np.nonzero(selected)
    spatial[rows, columns] = 1.0 / np.maximum(
        symmetric_distance[rows, columns], _MIN_DISTANCE
    )
    return production, attraction, spatial


def build_node_features(
    coordinates: np.ndarray,
    distance: np.ndarray,
    demand: np.ndarray,
    directional_bins: int = 36,
) -> np.ndarray:
    """Build min-max-normalized demand, distance, gravity, and direction features."""
    coords = np.asarray(coordinates, dtype=float)
    distances = _validate_square_matrix(distance, "distance", symmetric=True)
    demands = _validate_square_matrix(demand, "demand")
    size = distances.shape[0]
    if coords.shape != (size, 2) or not np.isfinite(coords).all():
        raise ValueError("coordinates must be a finite array with shape (nodes, 2)")
    if demands.shape != distances.shape:
        raise ValueError("demand and distance shapes must match")
    if isinstance(directional_bins, bool) or not isinstance(
        directional_bins, (int, np.integer)
    ) or directional_bins <= 0:
        raise ValueError("directional_bins must be a positive integer")

    demand_totals = demands.sum(axis=1) + demands.sum(axis=0)
    if demand_totals.sum() <= 0:
        raise ValueError("demand must have positive total flow")
    center = np.average(coords, axis=0, weights=demand_totals)
    center_distances = np.linalg.norm(coords - center, axis=1)
    angles = np.mod(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]), 2 * np.pi)
    directions = np.floor(angles * directional_bins / (2 * np.pi)).astype(int)
    directions = np.minimum(directions, directional_bins - 1)
    one_hot = np.eye(directional_bins, dtype=float)[directions]
    raw = np.column_stack(
        (demand_totals, distances.sum(axis=1), center_distances, one_hot)
    )
    return _min_max_normalize(raw)


def build_hub_feature_data(
    distance: np.ndarray,
    demand: np.ndarray,
    *,
    coordinates: np.ndarray | None = None,
    p: int = 3,
    alpha: float = 0.5,
    beta: float = 1.0,
    p_w: float = 0.8,
    p_c: float = 0.2,
    directional_bins: int = 36,
    seed: int = 0,
    time_limit: float = 1.0,
    label_method: LabelMethod = "heuristic",
    target_p_values: tuple[int, ...] | None = None,
    target_alpha_values: tuple[float, ...] | None = None,
) -> HubFeatureData:
    """Build graphs/features and label hubs with an existing project solver."""
    distances = _validate_square_matrix(distance, "distance", symmetric=True)
    demands = _validate_square_matrix(demand, "demand")
    if demands.shape != distances.shape:
        raise ValueError("demand and distance shapes must match")
    if not np.allclose(np.diag(distances), 0.0):
        raise ValueError("distance must have a zero diagonal")
    if not np.allclose(np.diag(demands), 0.0):
        raise ValueError("demand must have a zero diagonal")
    if demands.sum() <= 0:
        raise ValueError("demand must have positive total flow")
    size = distances.shape[0]
    if not 1 <= p <= size:
        raise ValueError("p must be between 1 and the number of nodes")
    if label_method not in {"exact", "heuristic"}:
        raise ValueError("label_method must be exact or heuristic")
    if label_method == "exact" and size > 8:
        raise ValueError("exact labels are limited to at most 8 nodes")

    coords = (
        classical_mds(distances)
        if coordinates is None
        else np.asarray(coordinates, dtype=float)
    )
    production, attraction, spatial = build_graphs(demands, distances, p_w, p_c)
    features = build_node_features(coords, distances, demands, directional_bins)
    normalized_demand = demands / demands.sum()
    scenario_flows = normalized_demand[None, :, :]
    probabilities = np.array([1.0])
    target_ps = (p,) if target_p_values is None else tuple(target_p_values)
    target_alphas = (
        (alpha,) if target_alpha_values is None else tuple(target_alpha_values)
    )
    if not target_ps or any(not 1 <= value <= size for value in target_ps):
        raise ValueError("target_p_values must contain values between 1 and node count")
    if not target_alphas or any(
        not np.isfinite(value) or value < 0 for value in target_alphas
    ):
        raise ValueError("target_alpha_values must contain finite nonnegative values")

    target_runs: list[dict[str, object]] = []
    selections = np.zeros(size, dtype=float)
    primary_solution: HubSolution | None = None
    primary_seed = seed
    target_grid = tuple(
        (target_p, target_alpha)
        for target_p in target_ps
        for target_alpha in target_alphas
    )
    for run_index, (target_p, target_alpha) in enumerate(target_grid):
        run_seed = seed + run_index
        solution = _solve_labels(
            distances,
            scenario_flows,
            probabilities,
            target_p,
            target_alpha,
            beta,
            run_seed,
            time_limit,
            label_method,
        )
        if not solution.hubs:
            raise RuntimeError(
                f"target label solver did not return a feasible solution: {solution.status}"
            )
        selections[list(solution.hubs)] += 1
        target_runs.append(
            {
                "p": target_p,
                "alpha": target_alpha,
                "seed": run_seed,
                "status": solution.status,
                "hubs": solution.hubs,
                "proven_optimal": solution.proven_optimal,
            }
        )
        if target_p == p and target_alpha == alpha and primary_solution is None:
            primary_solution = solution
            primary_seed = run_seed

    if primary_solution is None:
        primary_solution = _solve_labels(
            distances,
            scenario_flows,
            probabilities,
            p,
            alpha,
            beta,
            seed,
            time_limit,
            label_method,
        )
    if not primary_solution.hubs:
        raise RuntimeError(
            "primary label solver did not return a feasible solution: "
            f"{primary_solution.status}"
        )
    labels = np.zeros(size, dtype=np.int8)
    labels[list(primary_solution.hubs)] = 1
    scores = selections / len(target_runs)
    scores_are_exact = all(bool(run["proven_optimal"]) for run in target_runs)
    metadata: dict[str, object] = {
        "node_count": size,
        "p": p,
        "alpha": alpha,
        "beta": beta,
        "p_w": p_w,
        "p_c": p_c,
        "directional_bins": directional_bins,
        "seed": seed,
        "label_method": label_method,
        "label_seed": primary_seed,
        "label_status": primary_solution.status,
        "labels_proven_optimal": primary_solution.proven_optimal,
        "label_objective": primary_solution.objective,
        "target_grid": target_grid,
        "target_method": label_method,
        "target_runs": tuple(target_runs),
        "hub_scores_status": "exact" if scores_are_exact else "approximate",
        "all_target_solutions_proven_optimal": scores_are_exact,
        "coordinates_source": "provided" if coordinates is not None else "classical_mds",
    }
    return HubFeatureData(
        coordinates=coords.copy(),
        distance=distances.copy(),
        demand=demands.copy(),
        production_graph=production,
        attraction_graph=attraction,
        spatial_graph=spatial,
        features=features,
        hub_labels=labels,
        hub_scores=scores,
        metadata=metadata,
    )


def generate_synthetic_hub_data(
    *,
    seed: int,
    node_count: int = 25,
    p: int = 3,
    alpha: float = 0.5,
    beta: float = 1.0,
    p_w: float = 0.8,
    p_c: float = 0.2,
    directional_bins: int = 36,
    time_limit: float = 1.0,
    label_method: LabelMethod = "heuristic",
    target_p_values: tuple[int, ...] | None = None,
    target_alpha_values: tuple[float, ...] = (0.2, 0.5, 0.8),
    population_min: float = 1_000.0,
    population_max: float = 100_000.0,
    population_exponent: float = 2.0,
    gravity_scale: float = 1.0,
    minimum_distance: float = _MIN_DISTANCE,
) -> HubFeatureData:
    """Generate one seeded Euclidean gravity-model instance and ranking targets."""
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
        raise ValueError("node_count must be an integer of at least 2")
    if node_count < 2:
        raise ValueError("node_count must be an integer of at least 2")
    if not 0 < population_min < population_max:
        raise ValueError("population bounds must satisfy 0 < minimum < maximum")
    if not np.isfinite(population_exponent) or population_exponent <= 0:
        raise ValueError("population_exponent must be finite and positive")
    if not np.isfinite(gravity_scale) or gravity_scale <= 0:
        raise ValueError("gravity_scale must be finite and positive")
    if not np.isfinite(minimum_distance) or minimum_distance <= 0:
        raise ValueError("minimum_distance must be finite and positive")
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 100.0, size=(node_count, 2))
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.linalg.norm(offsets, axis=2)
    populations = _sample_bounded_power_law(
        rng,
        node_count,
        population_min,
        population_max,
        population_exponent,
    )
    safe_distance = np.maximum(distance, minimum_distance)
    demand = gravity_scale * np.outer(populations, populations) / np.square(safe_distance)
    np.fill_diagonal(demand, 0.0)
    score_ps = (
        tuple(value for value in (2, 3, 4, 5) if value <= node_count)
        if target_p_values is None
        else target_p_values
    )
    if not score_ps:
        score_ps = (p,)
    data = build_hub_feature_data(
        distance,
        demand,
        coordinates=coordinates,
        p=p,
        alpha=alpha,
        beta=beta,
        p_w=p_w,
        p_c=p_c,
        directional_bins=directional_bins,
        seed=seed,
        time_limit=time_limit,
        label_method=label_method,
        target_p_values=score_ps,
        target_alpha_values=target_alpha_values,
    )
    return replace(
        data,
        metadata={
            **data.metadata,
            "coordinates_source": "synthetic",
            "demand_generation": "gravity",
            "populations": tuple(float(value) for value in populations),
            "population_min": population_min,
            "population_max": population_max,
            "population_exponent": population_exponent,
            "gravity_scale": gravity_scale,
            "minimum_distance": minimum_distance,
        },
    )


def _solve_labels(
    distance: np.ndarray,
    scenario_flows: np.ndarray,
    probabilities: np.ndarray,
    p: int,
    alpha: float,
    beta: float,
    seed: int,
    time_limit: float,
    method: LabelMethod,
) -> HubSolution:
    if method == "exact":
        return solve_hub_model(
            build_hub_model(
                distance, scenario_flows, probabilities, p, alpha, beta
            ),
            time_limit=time_limit,
        )
    return solve_hub_heuristic(
        distance,
        scenario_flows,
        probabilities,
        p,
        alpha,
        beta,
        seed,
        time_limit,
    )


def _sample_bounded_power_law(
    rng: np.random.Generator,
    size: int,
    minimum: float,
    maximum: float,
    exponent: float,
) -> np.ndarray:
    uniform = rng.random(size)
    if np.isclose(exponent, 1.0):
        return minimum * np.power(maximum / minimum, uniform)
    power = 1.0 - exponent
    return np.power(
        np.power(minimum, power)
        + uniform * (np.power(maximum, power) - np.power(minimum, power)),
        1.0 / power,
    )


def _coverage_neighbors(values: np.ndarray, node: int, threshold: float) -> np.ndarray:
    candidates = np.flatnonzero((values > 0) & (np.arange(values.size) != node))
    if candidates.size == 0:
        return candidates
    order = np.lexsort((candidates, -values[candidates]))
    ranked = candidates[order]
    cutoff = int(np.searchsorted(np.cumsum(values[ranked]), threshold * values[ranked].sum()))
    return ranked[: cutoff + 1]


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    varying = ~np.isclose(span, 0.0)
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values, dtype=float),
        where=varying,
    )


def _validate_square_matrix(
    matrix: np.ndarray, name: str, *, symmetric: bool = False
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must contain finite, nonnegative values")
    if symmetric and not np.allclose(values, values.T):
        raise ValueError(f"{name} must be symmetric")
    return values


def _validate_feature_data(data: HubFeatureData) -> None:
    size = data.distance.shape[0]
    arrays = (
        data.coordinates,
        data.distance,
        data.demand,
        data.production_graph,
        data.attraction_graph,
        data.spatial_graph,
        data.features,
        data.hub_labels,
        data.hub_scores,
    )
    if data.coordinates.shape != (size, 2):
        raise ValueError("coordinates must have shape (nodes, 2)")
    if data.distance.shape != (size, size) or data.demand.shape != (size, size):
        raise ValueError("distance and demand must have shape (nodes, nodes)")
    if any(graph.shape != (size, size) for graph in arrays[3:6]):
        raise ValueError("graph matrices must have shape (nodes, nodes)")
    if data.features.ndim != 2 or data.features.shape[0] != size:
        raise ValueError("features must have one row per node")
    if data.hub_labels.shape != (size,):
        raise ValueError("hub_labels must have shape (nodes,)")
    if data.hub_scores.shape != (size,):
        raise ValueError("hub_scores must have shape (nodes,)")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("feature data arrays must contain finite values")
    if (data.distance < 0).any() or (data.demand < 0).any():
        raise ValueError("distance and demand must be nonnegative")
    if not np.allclose(data.distance, data.distance.T):
        raise ValueError("distance must be symmetric")
    if not np.allclose(np.diag(data.distance), 0.0) or not np.allclose(
        np.diag(data.demand), 0.0
    ):
        raise ValueError("distance and demand must have zero diagonals")
    if any((graph < 0).any() for graph in arrays[3:6]):
        raise ValueError("graph matrices must be nonnegative")
    if any(np.any(np.diag(graph)) for graph in arrays[3:6]):
        raise ValueError("graph matrices must have zero diagonals")
    if not np.allclose(data.spatial_graph, data.spatial_graph.T):
        raise ValueError("spatial_graph must be symmetric")
    if (data.features < 0).any() or (data.features > 1).any():
        raise ValueError("features must be normalized to [0, 1]")
    if not np.isin(data.hub_labels, (0, 1)).all():
        raise ValueError("hub_labels must be binary")
    if (data.hub_scores < 0).any() or (data.hub_scores > 1).any():
        raise ValueError("hub_scores must be in [0, 1]")
    expected_hubs = data.metadata.get("p")
    if expected_hubs is not None and int(expected_hubs) != int(data.hub_labels.sum()):
        raise ValueError("hub_labels must select metadata p hubs")
