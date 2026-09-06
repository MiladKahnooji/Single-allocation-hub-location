from pathlib import Path

import numpy as np
import pytest

from single_allocation_hub_location.gvns import (
    CandidateEvaluator,
    GVNSConfig,
    HubCandidate,
    _local_search,
    hub_swap_neighbors,
    initialize_candidate,
    node_reassignment_neighbors,
    shake_candidate,
    solve_dl_gvns,
    solve_gvns,
    validate_candidate,
)
from single_allocation_hub_location.gvns_experiments import (
    GVNSExperimentRecord,
    PairedGVNSConfig,
    comparison_summary,
    objective_improvement_percentage,
    run_paired_experiment,
    write_gvns_outputs,
)
from single_allocation_hub_location.ranker import CompactHubRanker, save_ranker_checkpoint


DISTANCE = np.array(
    [[0.0, 1.0, 3.0, 4.0], [1.0, 0.0, 2.0, 3.0], [3.0, 2.0, 0.0, 1.0], [4.0, 3.0, 1.0, 0.0]]
)
FLOWS = np.array(
    [
        [[0.0, 0.3, 0.0, 0.0], [0.1, 0.0, 0.2, 0.0], [0.0, 0.1, 0.0, 0.1], [0.1, 0.0, 0.1, 0.0]],
        [[0.0, 0.0, 0.2, 0.1], [0.2, 0.0, 0.0, 0.0], [0.1, 0.1, 0.0, 0.1], [0.0, 0.0, 0.2, 0.0]],
    ]
)
PROBABILITIES = np.array([0.5, 0.5])
CONFIG = GVNSConfig(p=2, alpha=0.5, beta=0.5, max_iterations=4, max_evaluations=16)


def test_candidate_validation_and_neighborhoods_preserve_feasibility() -> None:
    candidate = HubCandidate((0, 2), (0, 0, 2, 2))
    validate_candidate(candidate, node_count=4, p=2)
    with pytest.raises(ValueError, match="distinct hubs"):
        validate_candidate(HubCandidate((0, 0), (0, 0, 0, 0)), 4, 2)

    swaps = hub_swap_neighbors(candidate, DISTANCE)
    reassignments = node_reassignment_neighbors(candidate)

    assert swaps and reassignments
    for neighbor in swaps + reassignments:
        validate_candidate(neighbor, 4, 2)


def test_seeded_shaking_and_local_search_are_reproducible_and_nonworsening() -> None:
    initial = initialize_candidate(DISTANCE, 2, np.random.default_rng(4))
    first = shake_candidate(initial, DISTANCE, 2, np.random.default_rng(8), 2)
    second = shake_candidate(initial, DISTANCE, 2, np.random.default_rng(8), 2)
    assert first == second

    evaluator = CandidateEvaluator(DISTANCE, FLOWS, PROBABILITIES, CONFIG)
    before = evaluator.evaluate(initial)
    local, after = _local_search(initial, evaluator, None)
    validate_candidate(local, 4, 2)
    assert after <= before


def test_gvns_and_dl_gvns_are_seeded_and_rank_guided() -> None:
    baseline_first = solve_gvns(DISTANCE, FLOWS, PROBABILITIES, CONFIG, search_seed=6)
    baseline_second = solve_gvns(DISTANCE, FLOWS, PROBABILITIES, CONFIG, search_seed=6)
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    guided_first = solve_dl_gvns(DISTANCE, FLOWS, PROBABILITIES, CONFIG, 6, scores)
    guided_second = solve_dl_gvns(DISTANCE, FLOWS, PROBABILITIES, CONFIG, 6, scores)

    assert baseline_first.candidate == baseline_second.candidate
    assert baseline_first.objective == baseline_second.objective
    assert guided_first.candidate == guided_second.candidate
    assert guided_first.objective == guided_second.objective
    validate_candidate(guided_first.candidate, 4, 2)
    assert guided_first.ranker_identity == "provided_scores"
    assert baseline_first.ranker_identity is None


def test_rank_scores_control_guided_initialization_and_swap_priority() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    candidate = initialize_candidate(DISTANCE, 2, np.random.default_rng(1), scores)
    neighbors = hub_swap_neighbors(candidate, DISTANCE, scores)

    assert candidate.hubs == (1, 3)
    assert neighbors[0].hubs == (1, 2)


def test_paired_run_uses_identical_scenarios_and_budgets() -> None:
    baseline, guided = run_paired_experiment(
        PairedGVNSConfig(
            dataset="CAB25",
            p=2,
            scenario_seed=5,
            search_seed=7,
            max_iterations=2,
            max_evaluations=3,
        ),
        rank_scores=np.arange(25, dtype=float),
    )

    assert baseline.dataset == guided.dataset == "CAB25"
    assert baseline.scenario_count == guided.scenario_count == 100
    assert baseline.scenario_seed == guided.scenario_seed == 5
    assert baseline.search_seed == guided.search_seed == 7
    assert baseline.p == guided.p == 2
    assert baseline.evaluation_count <= 3
    assert guided.evaluation_count <= 3
    assert baseline.proven_optimal is guided.proven_optimal is False


def test_checkpoint_guided_cab25_run_is_reproducible(tmp_path: Path) -> None:
    """DL-GVNS loads a CPU checkpoint and repeats the same seeded result."""
    import torch

    torch.manual_seed(3)
    checkpoint = save_ranker_checkpoint(CompactHubRanker(39, hidden_dimension=4, layers=1), tmp_path / "ranker.pt")
    config = PairedGVNSConfig(
        dataset="CAB25", p=2, scenario_seed=4, search_seed=9, max_iterations=1, max_evaluations=2
    )
    _, first = run_paired_experiment(config, ranker_checkpoint=checkpoint)
    _, second = run_paired_experiment(config, ranker_checkpoint=checkpoint)

    assert first.hubs == second.hubs
    assert first.assignments == second.assignments
    assert first.objective == second.objective
    assert first.ranker_identity is not None and "ranker.pt" in first.ranker_identity


def test_comparison_metrics_and_serialization(tmp_path: Path) -> None:
    baseline = _record("gvns", 100.0, 2.0)
    guided = _record("dl_gvns", 80.0, 3.0)
    summary = comparison_summary([baseline, guided])
    paths = write_gvns_outputs([baseline, guided], tmp_path)

    assert objective_improvement_percentage(100.0, 80.0) == 20.0
    guided_summary = next(row for row in summary if row["method"] == "dl_gvns")
    assert guided_summary["dl_gvns_wins"] == 1
    assert guided_summary["ties"] == 0
    assert guided_summary["dl_gvns_losses"] == 0
    detail = paths["details"].read_text(encoding="utf-8")
    assert '"hubs"' in detail and '"assignments"' in detail
    assert all(path.stat().st_size > 0 for path in paths.values())


def _record(method: str, objective: float, runtime: float) -> GVNSExperimentRecord:
    return GVNSExperimentRecord(
        dataset="CAB25",
        node_count=4,
        method=method,
        scenario_seed=1,
        search_seed=2,
        p=2,
        alpha=0.5,
        beta=0.5,
        scenario_count=100,
        objective=objective,
        runtime=runtime,
        evaluation_count=3,
        iteration_count=2,
        hubs=(0, 2),
        assignments=(0, 0, 2, 2),
        status="iteration limit reached",
        proven_optimal=False,
        ranker_identity=None if method == "gvns" else "test",
    )
