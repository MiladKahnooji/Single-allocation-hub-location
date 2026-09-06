from pathlib import Path

import numpy as np
import pytest
import torch

from single_allocation_hub_location.ranker import (
    CompactHubRanker,
    RankerTrainingConfig,
    generate_ranker_dataset,
    load_ranker_checkpoint,
    normalize_graph_views,
    pairwise_ranking_accuracy,
    pairwise_ranking_loss,
    rank_hubs,
    split_instance_indices,
    top_p_recall,
    train_ranker,
)


@pytest.fixture(scope="module")
def tiny_training_data() -> tuple[RankerTrainingConfig, list]:
    config = RankerTrainingConfig(
        training_instance_count=4,
        validation_fraction=0.25,
        hidden_dimension=8,
        message_passing_layers=2,
        epochs=4,
        learning_rate=0.01,
        batch_size=2,
        patience=3,
        seed=17,
    )
    instances = generate_ranker_dataset(
        config,
        node_count=6,
        label_time_limit=0.05,
        target_p_values=(3,),
        target_alpha_values=(0.5,),
    )
    return config, instances


def test_message_aggregation_and_forward_shapes_are_finite(
    tiny_training_data: tuple[RankerTrainingConfig, list],
) -> None:
    _, instances = tiny_training_data
    instance = instances[0]
    features = torch.tensor(instance.features, dtype=torch.float32)
    graphs = torch.tensor(
        np.stack(
            (
                instance.production_graph,
                instance.attraction_graph,
                instance.spatial_graph,
            )
        ),
        dtype=torch.float32,
    )
    normalized = normalize_graph_views(graphs)
    model = CompactHubRanker(features.shape[1], hidden_dimension=8, layers=2)

    scores = model(features, graphs)
    batched_scores = model(
        features.unsqueeze(0).repeat(2, 1, 1),
        graphs.unsqueeze(0).repeat(2, 1, 1, 1),
    )

    assert normalized.shape == (3, 6, 6)
    assert torch.allclose(normalized.sum(dim=-1), torch.ones((3, 6)))
    assert scores.shape == (6,)
    assert batched_scores.shape == (2, 6)
    assert torch.isfinite(scores).all()
    with pytest.raises(ValueError, match="graphs must match"):
        model(features, graphs[:2])


def test_pairwise_loss_orders_unequal_targets_and_handles_no_pairs() -> None:
    targets = torch.tensor([1.0, 0.5, 0.0])
    ordered = pairwise_ranking_loss(torch.tensor([2.0, 1.0, 0.0]), targets)
    reversed_loss = pairwise_ranking_loss(torch.tensor([0.0, 1.0, 2.0]), targets)
    tied_scores = torch.tensor([0.2, 0.3, 0.4], requires_grad=True)
    no_pairs = pairwise_ranking_loss(tied_scores, torch.ones(3))

    assert ordered < reversed_loss
    assert no_pairs.item() == 0.0
    no_pairs.backward()
    assert torch.equal(tied_scores.grad, torch.zeros(3))


def test_ranking_metrics_match_known_order() -> None:
    targets = torch.tensor([1.0, 0.2, 0.8, 0.0])
    scores = torch.tensor([4.0, 2.0, 3.0, 1.0])

    assert pairwise_ranking_accuracy(scores, targets) == 1.0
    assert top_p_recall(scores, targets, p=2) == 1.0
    assert top_p_recall(torch.tensor([0.0, 4.0, 3.0, 2.0]), targets, p=2) == 0.5


def test_seeded_split_is_reproducible() -> None:
    first = split_instance_indices(10, validation_fraction=0.2, seed=9)
    second = split_instance_indices(10, validation_fraction=0.2, seed=9)
    different = split_instance_indices(10, validation_fraction=0.2, seed=10)

    assert first == second
    assert first != different
    assert len(first[0]) == 8
    assert len(first[1]) == 2
    assert set(first[0]).isdisjoint(first[1])


def test_tiny_training_checkpoint_and_inference_are_reproducible(
    tiny_training_data: tuple[RankerTrainingConfig, list], tmp_path: Path
) -> None:
    config, instances = tiny_training_data
    checkpoint = tmp_path / "ranker.pt"
    first = train_ranker(instances, config, checkpoint)
    second = train_ranker(instances, config)
    loaded = load_ranker_checkpoint(checkpoint)

    first_ranking = rank_hubs(first.model, instances[0])
    second_ranking = rank_hubs(second.model, instances[0])
    loaded_ranking = rank_hubs(loaded, instances[0])

    assert checkpoint.stat().st_size > 0
    assert np.isfinite(first.train_loss)
    assert np.isfinite(first.validation_loss)
    assert 0 <= first.pairwise_accuracy <= 1
    assert 0 <= first.top_p_recall <= 1
    assert 1 <= first.best_epoch <= first.epochs_trained <= config.epochs
    assert first_ranking.ranked_nodes == second_ranking.ranked_nodes
    assert first_ranking.ranked_nodes == loaded_ranking.ranked_nodes
    assert np.array_equal(first_ranking.scores, second_ranking.scores)
    assert np.array_equal(first_ranking.scores, loaded_ranking.scores)
    assert np.isfinite(loaded_ranking.scores).all()
