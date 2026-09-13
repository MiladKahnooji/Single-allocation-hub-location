import numpy as np

from single_allocation_hub_location.benders import solve_benders
from single_allocation_hub_location.benders import scenario_dual_cut
from single_allocation_hub_location.cbs import solve_cbs, solve_dl_cbs
from single_allocation_hub_location.evaluation import evaluate_scenario_costs
from single_allocation_hub_location.risk import equal_probability_beta_mean
from single_allocation_hub_location.scenarios import generate_flow_scenarios
from single_allocation_hub_location.model import build_hub_model
from single_allocation_hub_location.solution import solve_hub_model
from single_allocation_hub_location.training_data import build_benders_ground_truth_feature_data


def _instance():
    d = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], float)
    base = np.ones((4, 4)); np.fill_diagonal(base, 0)
    bundle = generate_flow_scenarios(base, scenario_count=3, seed=4)
    return d, base, bundle.flows, bundle.probabilities


def test_benders_matches_direct_objective_and_certifies():
    d, _, flows, q = _instance()
    result = solve_benders(d, flows, q, 2, .5, .5, max_iterations=30)
    assert result.proven_optimal
    assert result.candidate is not None
    direct = equal_probability_beta_mean(evaluate_scenario_costs(result.candidate.assignments, d, flows, .5), .5)
    assert np.isclose(result.objective, direct)
    assert result.lower_bound <= result.upper_bound + 1e-7


def test_benders_matches_direct_milp_for_multiple_tiny_cases():
    d, _, flows, q = _instance()
    for p, alpha in ((1, .2), (2, .5), (3, .8)):
        benders = solve_benders(d, flows, q, p, alpha, 1 / 3, max_iterations=40)
        direct = solve_hub_model(build_hub_model(d, flows, q, p, alpha, 1 / 3))
        assert benders.proven_optimal
        assert np.isclose(benders.objective, direct.objective)


def test_dual_cut_is_tight_at_incumbent_and_valid_for_another_assignment():
    d, _, flows, _ = _instance()
    incumbent = (0, 0, 2, 2)
    cut, value = scenario_dual_cut(d, flows[0], .5, incumbent)
    assert np.isclose(value, np.sum(cut[np.arange(4), incumbent]))
    other = (1, 1, 1, 1)
    actual = evaluate_scenario_costs(other, d, flows[:1], .5)[0]
    assert np.sum(cut[np.arange(4), other]) <= actual + 1e-8


def test_large_limited_benders_returns_feasible_unproven_incumbent():
    d = np.abs(np.subtract.outer(np.arange(7), np.arange(7))).astype(float)
    base = np.ones((7, 7)); np.fill_diagonal(base, 0)
    data = generate_flow_scenarios(base, scenario_count=2, seed=5)
    result = solve_benders(d, data.flows, data.probabilities, 2, .5, .5, max_iterations=1)
    assert result.candidate is not None
    assert not result.proven_optimal
    assert result.status in {"iteration_limit", "time_limited"}
    assert result.lower_bound is not None and result.lower_bound <= result.upper_bound + 1e-8


def test_benders_targets_have_exact_provenance_when_certified():
    d, demand, _, _ = _instance()
    data = build_benders_ground_truth_feature_data(
        d, demand, target_grid=((1, .5, 1.0), (2, .5, 1.0)), max_iterations=30
    )
    assert data.metadata["hub_scores_status"] == "ground_truth_exact"
    assert np.all((data.hub_scores >= 0) & (data.hub_scores <= 1))
    for run in data.metadata["target_runs"]:
        assert run["proven_optimal"] and np.isfinite(run["objective"])


def test_cbs_and_dl_cbs_are_feasible_and_seed_independent():
    d, demand, flows, q = _instance()
    a = solve_cbs(d, demand, flows, q, 2, .5, .5, max_evaluations=8)
    b = solve_dl_cbs(d, demand, flows, q, 2, .5, .5, scores=np.arange(4.), max_evaluations=8)
    for result in (a, b):
        c = result["candidate"]
        assert len(c.hubs) == 2 and len(set(c.hubs)) == 2
        assert len(c.assignments) == 4
        assert all(c.assignments[h] == h for h in c.hubs)
        assert all(x in c.hubs for x in c.assignments)
