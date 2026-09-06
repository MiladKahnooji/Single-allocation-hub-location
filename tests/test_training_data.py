from pathlib import Path

import numpy as np

from single_allocation_hub_location import (
    build_graphs,
    build_hub_feature_data,
    build_node_features,
    classical_mds,
    generate_synthetic_hub_data,
    load_matrix_pair,
)


def _distances(coordinates: np.ndarray) -> np.ndarray:
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(offsets, axis=2)


def test_classical_mds_is_deterministic_and_reconstructs_euclidean_distances() -> None:
    original = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    distance = _distances(original)

    first = classical_mds(distance)
    second = classical_mds(distance)

    assert first.shape == (4, 2)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert np.allclose(_distances(first), distance)


def test_graphs_follow_demand_coverage_and_nearest_node_semantics() -> None:
    demand = np.array([[0.0, 6.0, 4.0], [1.0, 0.0, 9.0], [8.0, 2.0, 0.0]])
    distance = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])

    production, attraction, spatial = build_graphs(
        demand, distance, p_w=0.6, p_c=0.5
    )

    assert np.array_equal(production, [[0, 6, 0], [0, 0, 9], [8, 0, 0]])
    assert np.array_equal(attraction, [[0, 0, 8], [6, 0, 0], [0, 9, 0]])
    assert np.array_equal(spatial, [[0, 1, 0], [1, 0, 0.5], [0, 0.5, 0]])
    assert np.array_equal(spatial, spatial.T)


def test_spatial_graph_handles_zero_distances_with_finite_symmetric_weights() -> None:
    demand = np.ones((3, 3)) - np.eye(3)
    distance = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

    _, _, spatial = build_graphs(demand, distance, p_c=0.5)

    assert np.isfinite(spatial).all()
    assert np.array_equal(spatial, spatial.T)
    assert np.all(np.diag(spatial) == 0)
    assert spatial[0, 1] == 1e12


def test_features_are_normalized_and_constant_columns_are_safe() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    distance = _distances(coordinates)
    demand = np.ones((3, 3)) - np.eye(3)

    features = build_node_features(
        coordinates, distance, demand, directional_bins=4
    )

    assert features.shape == (3, 7)
    assert np.isfinite(features).all()
    assert np.all((0 <= features) & (features <= 1))
    assert np.allclose(features[:, :2], 0.0)


def test_seeded_synthetic_data_is_reproducible_and_has_feasible_labels() -> None:
    first = generate_synthetic_hub_data(
        seed=7, node_count=6, p=2, time_limit=1.0, label_method="heuristic"
    )
    second = generate_synthetic_hub_data(
        seed=7, node_count=6, p=2, time_limit=1.0, label_method="heuristic"
    )

    for field in (
        "coordinates",
        "distance",
        "demand",
        "production_graph",
        "attraction_graph",
        "spatial_graph",
        "features",
        "hub_labels",
        "hub_scores",
    ):
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert first.coordinates.shape == (6, 2)
    assert first.distance.shape == first.demand.shape == (6, 6)
    assert np.allclose(first.distance, first.distance.T)
    assert np.all(np.diag(first.distance) == 0)
    assert np.all(np.diag(first.demand) == 0)
    assert np.array_equal(first.demand, first.demand.T)
    assert np.all(first.demand >= 0)
    assert first.hub_labels.sum() == 2
    assert np.all((0 <= first.hub_scores) & (first.hub_scores <= 1))
    assert first.metadata["label_method"] == "heuristic"
    assert first.metadata["coordinates_source"] == "synthetic"
    assert first.metadata["demand_generation"] == "gravity"
    assert first.metadata["hub_scores_status"] == "approximate"

    populations = np.asarray(first.metadata["populations"])
    expected = (
        float(first.metadata["gravity_scale"])
        * populations[0]
        * populations[1]
        / first.distance[0, 1] ** 2
    )
    assert first.demand[0, 1] == expected


def test_hub_scores_equal_selection_frequency_over_target_grid() -> None:
    data = generate_synthetic_hub_data(
        seed=13,
        node_count=6,
        p=2,
        directional_bins=4,
        time_limit=1.0,
        label_method="heuristic",
        target_p_values=(1, 2),
        target_alpha_values=(0.2, 0.8),
    )
    target_runs = data.metadata["target_runs"]
    counts = np.zeros(6)
    for run in target_runs:
        counts[list(run["hubs"])] += 1

    assert len(target_runs) == 4
    assert np.array_equal(data.hub_scores, counts / 4)
    assert [run["seed"] for run in target_runs] == [13, 14, 15, 16]


def test_tiny_exact_labels_are_marked_proven_optimal() -> None:
    data = generate_synthetic_hub_data(
        seed=3,
        node_count=4,
        p=1,
        directional_bins=4,
        time_limit=10.0,
        label_method="exact",
        target_p_values=(1,),
        target_alpha_values=(0.5,),
    )

    assert data.hub_labels.shape == (4,)
    assert data.hub_labels.sum() == 1
    assert data.metadata["labels_proven_optimal"] is True
    assert data.metadata["label_status"] == "Optimal Solution Found"
    assert data.metadata["hub_scores_status"] == "exact"
    assert np.array_equal(data.hub_scores, data.hub_labels)


def test_cab25_feature_pipeline_smoke() -> None:
    matrices = load_matrix_pair(
        Path("data/raw/wij_CAB25.xlsx"), Path("data/raw/C-CAB25.csv")
    )
    data = build_hub_feature_data(
        np.asarray(matrices.distance),
        np.asarray(matrices.flow),
        p=3,
        seed=11,
        time_limit=0.05,
        label_method="heuristic",
    )

    assert data.coordinates.shape == (25, 2)
    assert data.features.shape == (25, 39)
    assert data.hub_labels.sum() == 3
    assert data.hub_scores.shape == (25,)
    assert data.metadata["coordinates_source"] == "classical_mds"
    assert data.metadata["label_status"] == "Heuristic completed"
