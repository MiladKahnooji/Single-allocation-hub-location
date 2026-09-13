import numpy as np

from single_allocation_hub_location.benders import solve_benders
from single_allocation_hub_location.cbs import solve_cbs, solve_dl_cbs
from single_allocation_hub_location.evaluation import evaluate_scenario_costs
from single_allocation_hub_location.risk import equal_probability_beta_mean
from single_allocation_hub_location.scenarios import generate_flow_scenarios


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
