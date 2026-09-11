from pathlib import Path

import numpy as np
import pytest

from single_allocation_hub_location import (
    RunConfig,
    evaluate_risk_objective,
    load_run_configs,
    run_batch,
    run_experiment,
    select_method,
    solve_hub_heuristic,
)


DISTANCE = np.array(
    [
        [0.0, 1.0, 4.0, 6.0],
        [1.0, 0.0, 3.0, 5.0],
        [4.0, 3.0, 0.0, 2.0],
        [6.0, 5.0, 2.0, 0.0],
    ]
)
FLOWS = np.array(
    [
        [[0.0, 0.2, 0.1, 0.0], [0.1, 0.0, 0.1, 0.0], [0.0, 0.1, 0.0, 0.1], [0.1, 0.0, 0.1, 0.0]],
        [[0.0, 0.1, 0.0, 0.1], [0.2, 0.0, 0.0, 0.0], [0.1, 0.1, 0.0, 0.1], [0.0, 0.0, 0.2, 0.0]],
    ]
)
PROBABILITIES = np.array([0.5, 0.5])


def test_heuristic_is_deterministic_feasible_and_evaluated() -> None:
    first = solve_hub_heuristic(DISTANCE, FLOWS, PROBABILITIES, 2, 0.5, 0.5, 9, 1)
    second = solve_hub_heuristic(DISTANCE, FLOWS, PROBABILITIES, 2, 0.5, 0.5, 9, 1)

    assert first.hubs == second.hubs
    assert first.assignments == second.assignments
    assert first.objective == pytest.approx(second.objective)
    assert len(first.hubs) == len(set(first.hubs)) == 2
    assert len(first.assignments) == len(DISTANCE)
    assert all(assigned in first.hubs for assigned in first.assignments)
    assert all(first.assignments[hub] == hub for hub in first.hubs)
    assert first.objective == pytest.approx(
        evaluate_risk_objective(
            first.assignments, DISTANCE, FLOWS, PROBABILITIES, 0.5, 0.5
        )
    )
    assert not first.proven_optimal


@pytest.mark.parametrize(
    ("dataset", "scenario_count", "expected"),
    [
        ("CAB25", 5, "heuristic"),
        ("CAB25", 6, "heuristic"),
        ("AP100", 1, "heuristic"),
        ("AP150", 1, "heuristic"),
        ("AP200", 1, "heuristic"),
    ],
)
def test_auto_method_selection(dataset: str, scenario_count: int, expected: str) -> None:
    assert select_method(dataset, scenario_count, "auto") == expected


def test_json_batch_continues_after_invalid_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "runs.json"
    config_path.write_text(
        '{"runs": ['
        '{"dataset": "invalid", "p": 2, "alpha": 0.5, "beta": 0.5, '
        '"scenario_count": 1, "seed": 3, "time_limit": 0.01, "method": "heuristic"},'
        '{"dataset": "CAB25", "p": 2, "alpha": 0.5, "beta": 0.5, '
        '"scenario_count": 1, "seed": 3, "time_limit": 0.01, "method": "heuristic"}'
        ']}'
    )

    results = run_batch(load_run_configs(config_path))

    assert results[0].status == "error"
    assert results[0].error
    assert results[1].status == "Heuristic completed"
    assert results[1].error is None


@pytest.mark.parametrize(
    ("dataset", "node_count"),
    [("CAB25", 25), ("AP100", 100), ("AP150", 150), ("AP200", 200)],
)
def test_supplied_dataset_heuristic_smoke(dataset: str, node_count: int) -> None:
    result = run_experiment(
        RunConfig(dataset, 2, 0.5, 0.5, 1, 5, 0.01, "heuristic")
    )

    assert result.status == "Heuristic completed"
    assert result.node_count == node_count
    assert result.method_used == "heuristic"
    assert len(result.hubs) == 2
    assert len(result.assignments) == node_count
    assert result.runtime > 0
    assert result.error is None
