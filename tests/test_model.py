from itertools import combinations, product

import numpy as np
import pytest

from single_allocation_hub_location import (
    build_hub_model,
    conditional_beta_mean,
    evaluate_risk_objective,
    solve_hub_model,
)


DISTANCE = np.array(
    [
        [0.0, 2.0, 5.0],
        [2.0, 0.0, 3.0],
        [5.0, 3.0, 0.0],
    ]
)
FLOWS = np.array(
    [
        [[0.0, 0.5, 0.0], [0.0, 0.0, 0.2], [0.3, 0.0, 0.0]],
        [[0.0, 0.0, 0.4], [0.1, 0.0, 0.0], [0.0, 0.5, 0.0]],
    ]
)
PROBABILITIES = np.array([0.5, 0.5])


def test_conditional_beta_mean_for_known_costs() -> None:
    costs = np.array([1.0, 2.0, 10.0])
    probabilities = np.array([0.5, 0.25, 0.25])

    assert conditional_beta_mean(costs, probabilities, beta=0.5) == pytest.approx(6.0)


def test_conditional_beta_mean_uses_beta_as_worst_tail_fraction() -> None:
    costs = np.array([1.0, 2.0, 10.0])
    probabilities = np.array([0.5, 0.25, 0.25])

    assert conditional_beta_mean(costs, probabilities, beta=0.25) == pytest.approx(10.0)
    assert conditional_beta_mean(costs, probabilities, beta=1.0) == pytest.approx(3.5)


def test_smaller_beta_cannot_reduce_risk_value() -> None:
    costs = np.array([1.0, 2.0, 10.0])
    probabilities = np.array([0.5, 0.25, 0.25])
    values = [
        conditional_beta_mean(costs, probabilities, beta)
        for beta in (1.0, 0.5, 0.25)
    ]

    assert values == sorted(values)


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.1])
def test_conditional_beta_mean_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="0 < beta <= 1"):
        conditional_beta_mean(np.array([1.0]), np.array([1.0]), beta)


def test_exact_model_matches_brute_force() -> None:
    expected_objective, expected_assignments = _brute_force(p=1, alpha=0.5, beta=0.25)
    model = build_hub_model(DISTANCE, FLOWS, PROBABILITIES, p=1, alpha=0.5, beta=0.25)
    solution = solve_hub_model(model, time_limit=10)

    assert solution.status == "Optimal Solution Found"
    assert solution.proven_optimal
    assert solution.objective == pytest.approx(expected_objective)
    assert model.problem.objective.value() == pytest.approx(expected_objective)
    assert solution.assignments in expected_assignments
    assert len(solution.hubs) == 1
    assert all(solution.assignments.count(hub) >= 1 for hub in solution.hubs)
    assert all(assigned in solution.hubs for assigned in solution.assignments)


def test_exact_model_is_deterministic() -> None:
    first = solve_hub_model(
        build_hub_model(DISTANCE, FLOWS, PROBABILITIES, p=2, alpha=0.5, beta=0.5)
    )
    second = solve_hub_model(
        build_hub_model(DISTANCE, FLOWS, PROBABILITIES, p=2, alpha=0.5, beta=0.5)
    )

    assert first.hubs == second.hubs
    assert first.assignments == second.assignments
    assert first.objective == pytest.approx(second.objective)


def _brute_force(
    p: int, alpha: float, beta: float
) -> tuple[float, set[tuple[int, ...]]]:
    best = float("inf")
    best_assignments: set[tuple[int, ...]] = set()
    for hubs in combinations(range(len(DISTANCE)), p):
        for assignments in product(hubs, repeat=len(DISTANCE)):
            if any(assignments[hub] != hub for hub in hubs):
                continue
            objective = evaluate_risk_objective(
                assignments, DISTANCE, FLOWS, PROBABILITIES, alpha, beta
            )
            if objective < best - 1e-9:
                best = objective
                best_assignments = {assignments}
            elif objective == pytest.approx(best):
                best_assignments.add(assignments)
    return best, best_assignments
