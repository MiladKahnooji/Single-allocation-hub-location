"""Run selected exact or heuristic parameter configurations independently."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .data import load_matrix_pair
from .heuristic import solve_hub_heuristic
from .model import build_hub_model
from .scenarios import generate_flow_scenarios
from .solution import solve_hub_model

Method = Literal["exact", "heuristic", "auto"]

DATASETS = {
    "CAB25": ("wij_CAB25.xlsx", "C-CAB25.csv"),
    "AP100": ("wij_AP100.xlsx", "C_AP100.csv"),
    "AP150": ("wij_AP150.xlsx", "C_AP150.csv"),
    "AP200": ("wij_AP200.xlsx", "C_AP200.csv"),
}
ALLOWED_P = {2, 3, 4, 5}
ALLOWED_ALPHA = {0.2, 0.5, 0.8}
ALLOWED_BETA = {0.01, 0.1, 0.5, 0.8}


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    p: int
    alpha: float
    beta: float
    scenario_count: int = 100
    seed: int = 0
    time_limit: float = 60.0
    method: Method = "auto"


@dataclass(frozen=True)
class RunResult:
    dataset: str
    node_count: int | None
    method_requested: str
    method_used: str | None
    status: str
    proven_optimal: bool
    hubs: tuple[int, ...]
    assignments: tuple[int, ...]
    objective: float | None
    p: int
    alpha: float
    beta: float
    scenario_count: int
    seed: int
    time_limit: float
    runtime: float
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_method(dataset: str, scenario_count: int, requested: Method) -> str:
    """Resolve auto to scalable search for supplied CAB/AP dataset runs."""
    if requested != "auto":
        return requested
    return "heuristic"


def run_experiment(
    config: RunConfig, data_root: str | Path = Path("data/raw")
) -> RunResult:
    """Run one configuration and return an error record instead of raising."""
    started = time.perf_counter()
    method_used: str | None = None
    node_count: int | None = None
    try:
        _validate_config(config)
        method_used = select_method(config.dataset, config.scenario_count, config.method)
        if method_used == "exact" and not _exact_is_practical(
            config.dataset, config.scenario_count
        ):
            raise ValueError(
                "the exact binary route-selection model is only for tiny verification "
                "instances; use heuristic or auto for supplied CAB/AP datasets"
            )

        flow_name, distance_name = DATASETS[config.dataset]
        root = Path(data_root)
        matrices = load_matrix_pair(root / flow_name, root / distance_name)
        node_count = matrices.size
        scenarios = generate_flow_scenarios(
            matrices.flow, scenario_count=config.scenario_count, seed=config.seed
        )
        if method_used == "exact":
            solution = solve_hub_model(
                build_hub_model(
                    matrices.distance,
                    scenarios.flows,
                    scenarios.probabilities,
                    config.p,
                    config.alpha,
                    config.beta,
                ),
                time_limit=config.time_limit,
            )
        else:
            solution = solve_hub_heuristic(
                matrices.distance,
                scenarios.flows,
                scenarios.probabilities,
                config.p,
                config.alpha,
                config.beta,
                config.seed,
                config.time_limit,
            )
        return RunResult(
            dataset=config.dataset,
            node_count=node_count,
            method_requested=config.method,
            method_used=method_used,
            status=solution.status,
            proven_optimal=solution.proven_optimal if method_used == "exact" else False,
            hubs=solution.hubs,
            assignments=solution.assignments,
            objective=solution.objective,
            p=config.p,
            alpha=config.alpha,
            beta=config.beta,
            scenario_count=config.scenario_count,
            seed=config.seed,
            time_limit=config.time_limit,
            runtime=time.perf_counter() - started,
            error=None,
        )
    except Exception as error:
        return RunResult(
            dataset=config.dataset,
            node_count=node_count,
            method_requested=config.method,
            method_used=method_used,
            status="error",
            proven_optimal=False,
            hubs=(),
            assignments=(),
            objective=None,
            p=config.p,
            alpha=config.alpha,
            beta=config.beta,
            scenario_count=config.scenario_count,
            seed=config.seed,
            time_limit=config.time_limit,
            runtime=time.perf_counter() - started,
            error=str(error),
        )


def run_batch(configs: list[RunConfig], data_root: str | Path = Path("data/raw")) -> list[RunResult]:
    """Run configurations independently, preserving an error record for failures."""
    return [run_experiment(config, data_root) for config in configs]


def load_run_configs(path: str | Path) -> list[RunConfig]:
    """Read a JSON object with a ``runs`` list into configurations."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        document = json.load(config_file)
    if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
        raise ValueError("configuration JSON must contain a 'runs' list")
    return [RunConfig(**entry) for entry in document["runs"]]


def _exact_is_practical(dataset: str, scenario_count: int) -> bool:
    return False


def _validate_config(config: RunConfig) -> None:
    if config.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASETS)}")
    if config.p not in ALLOWED_P:
        raise ValueError("p must be one of: 2, 3, 4, 5")
    if config.alpha not in ALLOWED_ALPHA:
        raise ValueError("alpha must be one of: 0.2, 0.5, 0.8")
    if config.beta not in ALLOWED_BETA:
        raise ValueError("beta must be one of: 0.01, 0.1, 0.5, 0.8")
    if config.method not in {"exact", "heuristic", "auto"}:
        raise ValueError("method must be exact, heuristic, or auto")
    if config.scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    if config.time_limit <= 0:
        raise ValueError("time_limit must be positive")
