from itertools import combinations, product

import numpy as np
import pytest

import single_allocation_hub_location.evaluation as evaluation_module

from single_allocation_hub_location import (
    build_hub_model,
    conditional_beta_mean,
    equal_probability_beta_mean,
    evaluate_risk_objective,
    solve_hub_model,
    tail_scenario_count,
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


@pytest.mark.parametrize(("beta", "expected"), [(0.5, 50), (0.1, 10)])
def test_article_tail_count_for_100_scenarios(beta: float, expected: int) -> None:
    assert tail_scenario_count(100, beta) == expected


def test_equal_probability_article_form_matches_weighted_form_at_integer_tail_mass() -> None:
    costs = np.arange(1.0, 101.0)
    probabilities = np.full(100, 0.01)
    for beta in (0.5, 0.1):
        assert equal_probability_beta_mean(costs, beta) == pytest.approx(
            conditional_beta_mean(costs, probabilities, beta)
        )


def test_article_ceiling_rule_for_noninteger_tail_mass_is_explicit() -> None:
    costs = np.arange(1.0, 11.0)
    # ceil(0.25 * 10) = 3: average of 8, 9, and 10.
    assert tail_scenario_count(10, 0.25) == 3
    assert equal_probability_beta_mean(costs, 0.25) == pytest.approx(9.0)


def test_article_equal_probability_risk_is_monotone_for_smaller_tail_mass() -> None:
    costs = np.array([1.0, 2.0, 3.0, 10.0, 12.0])
    values = [equal_probability_beta_mean(costs, beta) for beta in (1.0, 0.6, 0.2)]
    assert values == sorted(values)


def test_risk_evaluator_passes_all_100_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flows = np.zeros((100, 2, 2))
    flows[:, 0, 1] = 1.0
    probabilities = np.arange(1.0, 101.0)
    probabilities /= probabilities.sum()
    received: dict[str, np.ndarray] = {}

    def capture(costs: np.ndarray, weights: np.ndarray, beta: float) -> float:
        received["costs"] = costs
        received["weights"] = weights
        return beta

    monkeypatch.setattr(evaluation_module, "conditional_beta_mean", capture)

    value = evaluation_module.evaluate_risk_objective(
        (0, 0), np.array([[0.0, 1.0], [1.0, 0.0]]), flows, probabilities, 0.5, 0.8
    )

    assert value == pytest.approx(0.8)
    assert received["costs"].shape == (100,)
    assert np.array_equal(received["weights"], probabilities)


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.1])
def test_conditional_beta_mean_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="0 < beta <= 1"):
        conditional_beta_mean(np.array([1.0]), np.array([1.0]), beta)


def test_exact_model_matches_brute_force() -> None:
    expected_objective, expected_assignments = _brute_force(p=1, alpha=0.5, beta=0.5)
    model = build_hub_model(DISTANCE, FLOWS, PROBABILITIES, p=1, alpha=0.5, beta=0.5)
    solution = solve_hub_model(model, time_limit=10)

    assert solution.status == "Optimal Solution Found"
    assert solution.proven_optimal
    assert solution.objective == pytest.approx(expected_objective)
    assert model.problem.objective.value() == pytest.approx(expected_objective)
    assert solution.assignments in expected_assignments
    assert len(solution.hubs) == 1
    assert all(solution.assignments.count(hub) >= 1 for hub in solution.hubs)
    assert all(assigned in solution.hubs for assigned in solution.assignments)


def test_exact_equal_probability_model_uses_ceiling_tail_count() -> None:
    flows = np.concatenate(
        [
            FLOWS,
            np.array([[[0.0, 0.1, 0.3], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        ]
    )
    probabilities = np.full(3, 1.0 / 3.0)
    model = build_hub_model(DISTANCE, flows, probabilities, p=1, alpha=0.5, beta=0.5)
    solution = solve_hub_model(model, time_limit=10)

    # beta * S = 1.5, so the article's ceiling form averages the worst two.
    assert tail_scenario_count(3, 0.5) == 2
    expected = equal_probability_beta_mean(solution.scenario_costs, 0.5)
    assert solution.objective == pytest.approx(expected)
    assert model.problem.objective.value() == pytest.approx(expected)
    assert solution.objective != pytest.approx(
        conditional_beta_mean(solution.scenario_costs, probabilities, 0.5)
    )


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


def test_exact_model_uses_assignment_diagonal_and_binary_route_selection() -> None:
    model = build_hub_model(DISTANCE, FLOWS, PROBABILITIES, p=1, alpha=0.5, beta=0.5)

    assert not hasattr(model, "hub")
    assert all(variable.cat == "Integer" for variable in model.assignment.values())
    assert all(variable.cat == "Integer" for variable in model.route_selection.values())
    constraints = model.problem.constraints
    assert "select_exactly_p_hubs_from_assignment_diagonal" in constraints
    assert any(name.startswith("select_one_route_") for name in constraints)


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
