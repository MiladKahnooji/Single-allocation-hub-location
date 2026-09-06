"""Lightweight graph and feature inputs for a later hub-ranking model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping

import numpy as np

from .heuristic import solve_hub_heuristic
from .model import build_hub_model
from .solution import solve_hub_model

LabelMethod = Literal["exact", "heuristic"]


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
    """Build outgoing-demand, incoming-demand, and directed nearest-node graphs."""
    demands = _validate_square_matrix(demand, "demand")
    distances = _validate_square_matrix(distance, "distance", symmetric=True)
    if demands.shape != distances.shape:
        raise ValueError("demand and distance shapes must match")
    if not 0 < p_w <= 1:
        raise ValueError("p_w must satisfy 0 < p_w <= 1")
    if not 0 < p_c <= 1:
        raise ValueError("p_c must satisfy 0 < p_c <= 1")

    size = demands.shape[0]
    production = np.zeros((size, size), dtype=np.int8)
    attraction = np.zeros((size, size), dtype=np.int8)
    for node in range(size):
        production[node, _coverage_neighbors(demands[node], node, p_w)] = 1
        attraction[node, _coverage_neighbors(demands[:, node], node, p_w)] = 1

    spatial = np.zeros((size, size), dtype=np.int8)
    neighbor_count = min(size - 1, max(1, int(np.ceil(p_c * (size - 1)))))
    indices = np.arange(size)
    for node in range(size):
        candidates = indices[indices != node]
        order = np.lexsort((candidates, distances[node, candidates]))
        spatial[node, candidates[order[:neighbor_count]]] = 1
    return production, attraction, spatial


def build_node_features(
    coordinates: np.ndarray,
    distance: np.ndarray,
    demand: np.ndarray,
    directional_bins: int = 4,
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
    directional_bins: int = 4,
    seed: int = 0,
    time_limit: float = 1.0,
    label_method: LabelMethod = "heuristic",
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
    if label_method == "exact":
        solution = solve_hub_model(
            build_hub_model(
                distances, scenario_flows, probabilities, p, alpha, beta
            ),
            time_limit=time_limit,
        )
    else:
        solution = solve_hub_heuristic(
            distances,
            scenario_flows,
            probabilities,
            p,
            alpha,
            beta,
            seed,
            time_limit,
        )
    if not solution.hubs:
        raise RuntimeError(f"label solver did not return a feasible solution: {solution.status}")
    labels = np.zeros(size, dtype=np.int8)
    labels[list(solution.hubs)] = 1
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
        "label_status": solution.status,
        "labels_proven_optimal": solution.proven_optimal,
        "label_objective": solution.objective,
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
    directional_bins: int = 4,
    time_limit: float = 1.0,
    label_method: LabelMethod = "heuristic",
) -> HubFeatureData:
    """Generate one seeded Euclidean instance and its hub labels."""
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
        raise ValueError("node_count must be an integer of at least 2")
    if node_count < 2:
        raise ValueError("node_count must be an integer of at least 2")
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 100.0, size=(node_count, 2))
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.linalg.norm(offsets, axis=2)
    demand = rng.poisson(10.0, size=(node_count, node_count)).astype(float)
    np.fill_diagonal(demand, 0.0)
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
    )
    return replace(
        data,
        metadata={**data.metadata, "coordinates_source": "synthetic"},
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
    if any(not np.isin(graph, (0, 1)).all() for graph in arrays[3:6]):
        raise ValueError("graph matrices must be binary")
    if any(np.any(np.diag(graph)) for graph in arrays[3:6]):
        raise ValueError("graph matrices must have zero diagonals")
    if (data.features < 0).any() or (data.features > 1).any():
        raise ValueError("features must be normalized to [0, 1]")
    if not np.isin(data.hub_labels, (0, 1)).all():
        raise ValueError("hub_labels must be binary")
    expected_hubs = data.metadata.get("p")
    if expected_hubs is not None and int(expected_hubs) != int(data.hub_labels.sum()):
        raise ValueError("hub_labels must select metadata p hubs")
