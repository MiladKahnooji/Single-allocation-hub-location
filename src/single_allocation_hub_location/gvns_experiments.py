"""Fair paired GVNS/DL-GVNS runs and compact comparison exports."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np

from .data import load_matrix_pair
from .experiments import DATASETS
from .gvns import GVNSConfig, GVNSResult, load_rank_scores, solve_dl_gvns, solve_gvns
from .scenarios import generate_flow_scenarios

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class PairedGVNSConfig:
    """All shared parameters for one fair baseline/guided comparison."""

    dataset: str
    p: int = 3
    alpha: float = 0.5
    beta: float = 0.5
    scenario_count: int = 100
    scenario_seed: int = 0
    search_seed: int = 0
    max_iterations: int = 4
    max_evaluations: int = 12
    time_limit: float | None = None


@dataclass(frozen=True)
class GVNSExperimentRecord:
    """Serializable detailed record for one method in a paired run."""

    dataset: str
    node_count: int
    method: str
    scenario_seed: int
    search_seed: int
    p: int
    alpha: float
    beta: float
    scenario_count: int
    objective: float
    runtime: float
    evaluation_count: int
    iteration_count: int
    hubs: tuple[int, ...]
    assignments: tuple[int, ...]
    status: str
    proven_optimal: bool
    ranker_identity: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_paired_experiment(
    config: PairedGVNSConfig,
    *,
    data_root: str | Path = Path("data/raw"),
    ranker_checkpoint: str | Path | None = None,
    rank_scores: np.ndarray | None = None,
) -> tuple[GVNSExperimentRecord, GVNSExperimentRecord]:
    """Run both methods on the same generated scenarios and search budget."""
    if config.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASETS)}")
    if config.scenario_count != 100:
        raise ValueError("paired GVNS comparisons require exactly 100 scenarios")
    flow_name, distance_name = DATASETS[config.dataset]
    matrices = load_matrix_pair(Path(data_root) / flow_name, Path(data_root) / distance_name)
    distance = np.asarray(matrices.distance)
    demand = np.asarray(matrices.flow)
    scenarios = generate_flow_scenarios(
        demand, scenario_count=config.scenario_count, seed=config.scenario_seed
    )
    search_config = GVNSConfig(
        p=config.p,
        alpha=config.alpha,
        beta=config.beta,
        max_iterations=config.max_iterations,
        max_evaluations=config.max_evaluations,
        time_limit=config.time_limit,
    )
    if rank_scores is None:
        if ranker_checkpoint is None:
            raise ValueError("DL-GVNS requires rank_scores or a ranker_checkpoint")
        rank_scores, ranker_identity = load_rank_scores(distance, demand, ranker_checkpoint)
    else:
        ranker_identity = "provided_scores"
    baseline = solve_gvns(
        distance, scenarios.flows, scenarios.probabilities, search_config, config.search_seed
    )
    guided = solve_dl_gvns(
        distance,
        scenarios.flows,
        scenarios.probabilities,
        search_config,
        config.search_seed,
        rank_scores,
        ranker_identity,
    )
    return (
        _record(config, distance.shape[0], baseline),
        _record(config, distance.shape[0], guided),
    )


def run_small_grid(
    *,
    datasets: Sequence[str] = ("CAB25", "AP100", "AP150", "AP200"),
    seeds: Sequence[int] = (0, 1),
    p: int = 3,
    alpha: float = 0.5,
    beta: float = 0.5,
    max_iterations: int = 4,
    max_evaluations: int = 12,
    time_limit: float | None = None,
    ranker_checkpoint: str | Path | None = None,
    data_root: str | Path = Path("data/raw"),
) -> list[GVNSExperimentRecord]:
    """Run a deliberately small paired grid; callers choose a ranker checkpoint."""
    records: list[GVNSExperimentRecord] = []
    for dataset in datasets:
        for seed in seeds:
            pair = run_paired_experiment(
                PairedGVNSConfig(
                    dataset=dataset,
                    p=p,
                    alpha=alpha,
                    beta=beta,
                    scenario_seed=seed,
                    search_seed=seed,
                    max_iterations=max_iterations,
                    max_evaluations=max_evaluations,
                    time_limit=time_limit,
                ),
                data_root=data_root,
                ranker_checkpoint=ranker_checkpoint,
            )
            records.extend(pair)
    return records


def train_synthetic_ranker_checkpoint(
    checkpoint_path: str | Path,
    *,
    training_instance_count: int = 6,
    seed: int = 0,
) -> str:
    """Train only on small synthetic data and save a CPU ranker checkpoint."""
    from .ranker import RankerTrainingConfig, generate_ranker_dataset, train_ranker

    config = RankerTrainingConfig(
        training_instance_count=training_instance_count,
        validation_fraction=0.25,
        hidden_dimension=16,
        message_passing_layers=1,
        epochs=8,
        learning_rate=0.01,
        batch_size=2,
        patience=3,
        seed=seed,
    )
    data = generate_ranker_dataset(
        config,
        node_count=8,
        label_time_limit=0.05,
        target_p_values=(2, 3),
        target_alpha_values=(0.5,),
    )
    train_ranker(data, config, checkpoint_path)
    return f"synthetic_training:{training_instance_count}:seed={seed}"


def objective_improvement_percentage(gvns_objective: float, dl_gvns_objective: float) -> float:
    """Return positive improvement when guided search lowers a minimization objective."""
    if not np.isfinite(gvns_objective) or not np.isfinite(dl_gvns_objective):
        raise ValueError("objectives must be finite")
    if gvns_objective == 0:
        return 0.0 if dl_gvns_objective == 0 else float("-inf")
    return 100.0 * (gvns_objective - dl_gvns_objective) / gvns_objective


def comparison_summary(records: Iterable[GVNSExperimentRecord]) -> list[dict[str, object]]:
    """Aggregate mean objectives/runtimes and DL-GVNS win/tie/loss counts."""
    grouped: dict[str, list[GVNSExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset, []).append(record)
    summaries: list[dict[str, object]] = []
    for dataset in sorted(grouped):
        values = grouped[dataset]
        by_method = {method: [item for item in values if item.method == method] for method in ("gvns", "dl_gvns")}
        pairs = _paired_records(values)
        wins = ties = losses = 0
        improvements: list[float] = []
        for baseline, guided in pairs:
            improvement = objective_improvement_percentage(baseline.objective, guided.objective)
            improvements.append(improvement)
            if guided.objective < baseline.objective - 1e-12:
                wins += 1
            elif guided.objective > baseline.objective + 1e-12:
                losses += 1
            else:
                ties += 1
        for method, method_records in by_method.items():
            if method_records:
                summaries.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "run_count": len(method_records),
                        "mean_objective": float(np.mean([item.objective for item in method_records])),
                        "mean_runtime": float(np.mean([item.runtime for item in method_records])),
                        "dl_gvns_wins": wins if method == "dl_gvns" else "",
                        "ties": ties if method == "dl_gvns" else "",
                        "dl_gvns_losses": losses if method == "dl_gvns" else "",
                        "mean_improvement_percent": float(np.mean(improvements)) if improvements else "",
                    }
                )
    return summaries


def write_gvns_outputs(
    records: Sequence[GVNSExperimentRecord], output_dir: str | Path
) -> dict[str, Path]:
    """Write detailed JSON, aggregate CSV, and objective/runtime plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "details": destination / "gvns_results.json",
        "summary": destination / "gvns_summary.csv",
        "objective_plot": destination / "gvns_objective_comparison.png",
        "runtime_plot": destination / "gvns_runtime_comparison.png",
    }
    paths["details"].write_text(
        json.dumps([record.to_dict() for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = comparison_summary(records)
    fields = (
        "dataset", "method", "run_count", "mean_objective", "mean_runtime",
        "dl_gvns_wins", "ties", "dl_gvns_losses", "mean_improvement_percent",
    )
    with paths["summary"].open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    _comparison_plot(summary, "mean_objective", "Mean objective", paths["objective_plot"])
    _comparison_plot(summary, "mean_runtime", "Mean runtime (seconds)", paths["runtime_plot"])
    return paths


def _record(config: PairedGVNSConfig, node_count: int, result: GVNSResult) -> GVNSExperimentRecord:
    return GVNSExperimentRecord(
        dataset=config.dataset,
        node_count=node_count,
        method=result.method,
        scenario_seed=config.scenario_seed,
        search_seed=config.search_seed,
        p=config.p,
        alpha=config.alpha,
        beta=config.beta,
        scenario_count=config.scenario_count,
        objective=result.objective,
        runtime=result.runtime,
        evaluation_count=result.evaluation_count,
        iteration_count=result.iteration_count,
        hubs=result.candidate.hubs,
        assignments=result.candidate.assignments,
        status=result.status,
        proven_optimal=False,
        ranker_identity=result.ranker_identity,
    )


def _paired_records(
    records: Sequence[GVNSExperimentRecord],
) -> list[tuple[GVNSExperimentRecord, GVNSExperimentRecord]]:
    grouped: dict[tuple[object, ...], dict[str, GVNSExperimentRecord]] = {}
    for record in records:
        key = (
            record.dataset, record.scenario_seed, record.search_seed, record.p,
            record.alpha, record.beta, record.scenario_count,
        )
        grouped.setdefault(key, {})[record.method] = record
    return [
        (methods["gvns"], methods["dl_gvns"])
        for methods in grouped.values()
        if {"gvns", "dl_gvns"} <= methods.keys()
    ]


def _comparison_plot(summary: Sequence[dict[str, object]], field: str, label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(6, len(summary) * 1.1), 4))
    labels = [f"{row['dataset']}\n{row['method']}" for row in summary]
    values = [float(row[field]) for row in summary]
    axis.bar(labels, values, color=["tab:blue" if row["method"] == "gvns" else "tab:green" for row in summary])
    axis.set_title(label + " by dataset and method")
    axis.set_ylabel(label)
    axis.set_xlabel("Dataset and method")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
