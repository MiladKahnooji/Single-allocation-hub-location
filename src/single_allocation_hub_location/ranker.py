"""Compact direct-PyTorch node ranker for the Issue #12 feature contract."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .training_data import HubFeatureData, generate_synthetic_hub_data


@dataclass(frozen=True)
class RankerTrainingConfig:
    """CPU-first settings for a small reproducible training run."""

    training_instance_count: int = 20
    validation_fraction: float = 0.2
    hidden_dimension: int = 32
    message_passing_layers: int = 2
    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 4
    patience: int = 8
    seed: int = 0


@dataclass(frozen=True)
class TrainingResult:
    """Trained model, split, stopping state, and final ranking metrics."""

    model: CompactHubRanker
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    epochs_trained: int
    best_epoch: int
    train_loss: float
    validation_loss: float
    pairwise_accuracy: float
    top_p_recall: float


@dataclass(frozen=True)
class RankingResult:
    """Deterministic descending node order and corresponding raw scores."""

    ranked_nodes: tuple[int, ...]
    scores: np.ndarray


class CompactHubRanker(nn.Module):
    """Three-view aggregation, GRU updates, jumping knowledge, and decoding."""

    def __init__(self, input_dimension: int, hidden_dimension: int = 32, layers: int = 2):
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0 or layers <= 0:
            raise ValueError("model dimensions and layers must be positive")
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.layers = layers
        self.input_projection = nn.Linear(input_dimension, hidden_dimension)
        self.message_projections = nn.ModuleList(
            nn.Linear(3 * hidden_dimension, hidden_dimension) for _ in range(layers)
        )
        self.gru_cells = nn.ModuleList(
            nn.GRUCell(hidden_dimension, hidden_dimension) for _ in range(layers)
        )
        self.decoder = nn.Linear((layers + 1) * hidden_dimension, 1)

    def forward(self, features: Tensor, graphs: Tensor) -> Tensor:
        """Return one finite score per node for unbatched or batched inputs."""
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
            graphs = graphs.unsqueeze(0)
        _validate_model_tensors(features, graphs, self.input_dimension)

        normalized = normalize_graph_views(graphs)
        hidden = torch.tanh(self.input_projection(features))
        representations = [hidden]
        for projection, cell in zip(self.message_projections, self.gru_cells):
            messages = torch.cat(
                [torch.bmm(normalized[:, view], hidden) for view in range(3)], dim=-1
            )
            message_input = torch.tanh(projection(messages))
            shape = hidden.shape
            hidden = cell(
                message_input.reshape(-1, self.hidden_dimension),
                hidden.reshape(-1, self.hidden_dimension),
            ).reshape(shape)
            representations.append(hidden)
        scores = self.decoder(torch.cat(representations, dim=-1)).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise RuntimeError("ranker produced non-finite scores")
        return scores.squeeze(0) if unbatched else scores


def normalize_graph_views(graphs: Tensor) -> Tensor:
    """Add self loops and row-normalize each weighted graph view."""
    if graphs.ndim not in {3, 4}:
        raise ValueError("graphs must have shape (3, nodes, nodes) or batched equivalent")
    unbatched = graphs.ndim == 3
    values = graphs.unsqueeze(0) if unbatched else graphs
    if values.shape[1] != 3 or values.shape[2] != values.shape[3]:
        raise ValueError("graphs must contain three square graph views")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("graphs must contain finite, nonnegative weights")
    if not values.is_floating_point():
        raise ValueError("graphs must use a floating-point dtype")
    nodes = values.shape[-1]
    identity = torch.eye(nodes, dtype=values.dtype, device=values.device).view(1, 1, nodes, nodes)
    with_self = values + identity
    normalized = with_self / with_self.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(values.dtype).eps
    )
    return normalized.squeeze(0) if unbatched else normalized


def pairwise_ranking_loss(scores: Tensor, targets: Tensor) -> Tensor:
    """Logistic pairwise loss over node pairs with unequal target scores."""
    if scores.shape != targets.shape or scores.ndim not in {1, 2}:
        raise ValueError("scores and targets must have matching node-vector shapes")
    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
        raise ValueError("scores and targets must be finite")
    score_rows = scores.unsqueeze(0) if scores.ndim == 1 else scores
    target_rows = targets.unsqueeze(0) if targets.ndim == 1 else targets
    losses: list[Tensor] = []
    for predicted, expected in zip(score_rows, target_rows):
        target_difference = expected[:, None] - expected[None, :]
        valid = torch.triu(target_difference != 0, diagonal=1)
        if torch.any(valid):
            score_difference = predicted[:, None] - predicted[None, :]
            direction = torch.sign(target_difference[valid])
            losses.append(F.softplus(-direction * score_difference[valid]).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def pairwise_ranking_accuracy(scores: Tensor, targets: Tensor) -> float:
    """Return the fraction of unequal target pairs ordered correctly."""
    if scores.shape != targets.shape or scores.ndim != 1:
        raise ValueError("scores and targets must be matching node vectors")
    target_difference = targets[:, None] - targets[None, :]
    valid = torch.triu(target_difference != 0, diagonal=1)
    if not torch.any(valid):
        return 0.0
    score_difference = scores[:, None] - scores[None, :]
    correct = score_difference[valid] * target_difference[valid] > 0
    return float(correct.float().mean())


def top_p_recall(scores: Tensor, targets: Tensor, p: int) -> float:
    """Return overlap between predicted and target top-p nodes divided by p."""
    if scores.shape != targets.shape or scores.ndim != 1:
        raise ValueError("scores and targets must be matching node vectors")
    if not 1 <= p <= scores.numel():
        raise ValueError("p must be between 1 and the number of nodes")
    predicted = set(torch.argsort(scores, descending=True, stable=True)[:p].tolist())
    expected = set(torch.argsort(targets, descending=True, stable=True)[:p].tolist())
    return len(predicted & expected) / p


def split_instance_indices(
    instance_count: int, validation_fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Create a reproducible nonempty train/validation split."""
    if instance_count < 2:
        raise ValueError("at least two instances are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must satisfy 0 < value < 1")
    validation_count = min(
        instance_count - 1, max(1, int(round(instance_count * validation_fraction)))
    )
    order = np.random.default_rng(seed).permutation(instance_count)
    validation = tuple(int(value) for value in order[:validation_count])
    training = tuple(int(value) for value in order[validation_count:])
    return training, validation


def generate_ranker_dataset(
    config: RankerTrainingConfig,
    *,
    node_count: int = 25,
    label_time_limit: float = 0.05,
    target_p_values: tuple[int, ...] = (2, 3, 4, 5),
    target_alpha_values: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> list[HubFeatureData]:
    """Generate a small in-memory seeded dataset; no artifacts are written."""
    _validate_training_config(config)
    return [
        generate_synthetic_hub_data(
            seed=config.seed + index,
            node_count=node_count,
            time_limit=label_time_limit,
            target_p_values=tuple(p for p in target_p_values if p <= node_count),
            target_alpha_values=target_alpha_values,
        )
        for index in range(config.training_instance_count)
    ]


def train_ranker(
    instances: Sequence[HubFeatureData],
    config: RankerTrainingConfig,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train deterministically on CPU with early stopping."""
    _validate_training_config(config)
    if len(instances) != config.training_instance_count:
        raise ValueError("instances must match training_instance_count")
    tensors = [_instance_tensors(instance) for instance in instances]
    feature_dimension = tensors[0][0].shape[1]
    node_count = tensors[0][0].shape[0]
    if any(item[0].shape != (node_count, feature_dimension) for item in tensors):
        raise ValueError("all instances must have matching node and feature dimensions")

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = CompactHubRanker(
        feature_dimension, config.hidden_dimension, config.message_passing_layers
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_indices, validation_indices = split_instance_indices(
        len(instances), config.validation_fraction, config.seed
    )
    rng = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_train_loss = float("inf")
    stale_epochs = 0
    final_train_loss = float("nan")
    epochs_trained = 0

    for epoch in range(config.epochs):
        model.train()
        shuffled = np.asarray(train_indices)[rng.permutation(len(train_indices))]
        batch_losses: list[float] = []
        for start in range(0, len(shuffled), config.batch_size):
            indices = shuffled[start : start + config.batch_size]
            features, graphs, targets = _stack_batch(tensors, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_ranking_loss(model(features, graphs), targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        final_train_loss = float(np.mean(batch_losses))
        validation_loss = _mean_loss(model, tensors, validation_indices)
        epochs_trained = epoch + 1
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_train_loss = final_train_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    validation_loss = _mean_loss(model, tensors, validation_indices)
    accuracy, recall = _mean_metrics(model, tensors, instances, validation_indices)
    if checkpoint_path is not None:
        save_ranker_checkpoint(model, checkpoint_path, {"training": asdict(config)})
    return TrainingResult(
        model=model,
        train_indices=train_indices,
        validation_indices=validation_indices,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        train_loss=best_train_loss,
        validation_loss=validation_loss,
        pairwise_accuracy=accuracy,
        top_p_recall=recall,
    )


def save_ranker_checkpoint(
    model: CompactHubRanker,
    path: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Save model dimensions, CPU weights, and optional primitive metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {
                "input_dimension": model.input_dimension,
                "hidden_dimension": model.hidden_dimension,
                "layers": model.layers,
            },
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_ranker_checkpoint(path: str | Path) -> CompactHubRanker:
    """Load a checkpoint onto CPU and return an evaluation-mode model."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    settings = checkpoint["model"]
    model = CompactHubRanker(
        int(settings["input_dimension"]),
        int(settings["hidden_dimension"]),
        int(settings["layers"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def rank_hubs(model: CompactHubRanker, instance: HubFeatureData) -> RankingResult:
    """Run deterministic CPU inference and rank nodes by descending score."""
    features, graphs, _ = _instance_tensors(instance)
    model = model.cpu().eval()
    with torch.no_grad():
        scores = model(features, graphs).cpu().numpy().astype(float)
    indices = np.arange(scores.size)
    order = np.lexsort((indices, -scores))
    return RankingResult(tuple(int(value) for value in order), scores)


def _instance_tensors(instance: HubFeatureData) -> tuple[Tensor, Tensor, Tensor]:
    features = torch.as_tensor(instance.features, dtype=torch.float32)
    graphs = torch.as_tensor(
        np.stack(
            (
                instance.production_graph,
                instance.attraction_graph,
                instance.spatial_graph,
            )
        ),
        dtype=torch.float32,
    )
    targets = torch.as_tensor(instance.hub_scores, dtype=torch.float32)
    return features, graphs, targets


def _stack_batch(
    tensors: Sequence[tuple[Tensor, Tensor, Tensor]], indices: Sequence[int]
) -> tuple[Tensor, Tensor, Tensor]:
    selected = [tensors[int(index)] for index in indices]
    return tuple(torch.stack(items) for items in zip(*selected))  # type: ignore[return-value]


def _mean_loss(
    model: CompactHubRanker,
    tensors: Sequence[tuple[Tensor, Tensor, Tensor]],
    indices: Sequence[int],
) -> float:
    model.eval()
    with torch.no_grad():
        features, graphs, targets = _stack_batch(tensors, indices)
        return float(pairwise_ranking_loss(model(features, graphs), targets))


def _mean_metrics(
    model: CompactHubRanker,
    tensors: Sequence[tuple[Tensor, Tensor, Tensor]],
    instances: Sequence[HubFeatureData],
    indices: Sequence[int],
) -> tuple[float, float]:
    accuracies: list[float] = []
    recalls: list[float] = []
    model.eval()
    with torch.no_grad():
        for index in indices:
            features, graphs, targets = tensors[index]
            scores = model(features, graphs)
            accuracies.append(pairwise_ranking_accuracy(scores, targets))
            recalls.append(top_p_recall(scores, targets, int(instances[index].metadata["p"])))
    return float(np.mean(accuracies)), float(np.mean(recalls))


def _validate_model_tensors(features: Tensor, graphs: Tensor, input_dimension: int) -> None:
    if features.ndim != 3 or graphs.ndim != 4:
        raise ValueError("features and graphs must be batched tensors")
    if features.shape[-1] != input_dimension:
        raise ValueError("feature dimension does not match the model")
    if graphs.shape[:2] != (features.shape[0], 3) or graphs.shape[2:] != (
        features.shape[1],
        features.shape[1],
    ):
        raise ValueError("graphs must match the batch and node dimensions")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")


def _validate_training_config(config: RankerTrainingConfig) -> None:
    integer_values = (
        config.training_instance_count,
        config.hidden_dimension,
        config.message_passing_layers,
        config.epochs,
        config.batch_size,
        config.patience,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in integer_values
    ):
        raise ValueError("training counts and dimensions must be positive integers")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ValueError("seed must be an integer")
    if config.training_instance_count < 2:
        raise ValueError("training_instance_count must be at least 2")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must satisfy 0 < value < 1")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
