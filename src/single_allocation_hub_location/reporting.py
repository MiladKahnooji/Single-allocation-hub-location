"""Export experiment records and small comparison plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

from .experiments import RunResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SUMMARY_FIELDS = (
    "dataset",
    "node_count",
    "method_requested",
    "method_used",
    "status",
    "proven_optimal",
    "objective",
    "p",
    "alpha",
    "beta",
    "scenario_count",
    "seed",
    "time_limit",
    "runtime",
    "error",
)


def write_results(results: list[RunResult], output_dir: str | Path) -> dict[str, Path]:
    """Write summary, detailed solutions, and comparison plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": destination / "summary.csv",
        "solutions": destination / "solutions.json",
        "objective_plot": destination / "objective_comparison.png",
        "runtime_plot": destination / "runtime_comparison.png",
    }
    _write_summary(results, paths["summary"])
    paths["solutions"].write_text(
        json.dumps([result.to_dict() for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    _write_objective_plot(results, paths["objective_plot"])
    _write_runtime_plot(results, paths["runtime_plot"])
    return paths


def _write_summary(results: list[RunResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            record = result.to_dict()
            writer.writerow({field: record[field] for field in SUMMARY_FIELDS})


def _write_objective_plot(results: list[RunResult], path: Path) -> None:
    successful = [result for result in results if result.error is None and result.objective is not None]
    figure, axis = plt.subplots(figsize=(max(6, len(successful) * 1.2), 4))
    if successful:
        labels = [_label(result) for result in successful]
        objectives = [result.objective for result in successful]
        axis.bar(labels, objectives, color="tab:blue")
        if min(objectives) > 0 and max(objectives) / min(objectives) > 10:
            axis.set_yscale("log")
            axis.set_ylabel("Conditional beta-mean objective (log scale)")
        axis.tick_params(axis="x", rotation=30)
    else:
        axis.text(0.5, 0.5, "No successful runs", ha="center", va="center")
        axis.set_xticks([])
    axis.set_title("Objective comparison for successful runs")
    axis.set_xlabel("Run")
    if not successful or axis.get_yscale() != "log":
        axis.set_ylabel("Conditional beta-mean objective")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _write_runtime_plot(results: list[RunResult], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(6, len(results) * 1.2), 4))
    labels = [_label(result) for result in results]
    axis.bar(labels, [result.runtime for result in results], color="tab:orange")
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("Runtime comparison")
    axis.set_xlabel("Run")
    axis.set_ylabel("Runtime (seconds)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _label(result: RunResult) -> str:
    method = result.method_used or result.method_requested
    return f"{result.dataset}\n{method}, p={result.p}"
