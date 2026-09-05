from pathlib import Path

import numpy as np
import pytest

import single_allocation_hub_location.scenarios as scenario_module
from single_allocation_hub_location import generate_flow_scenarios, load_matrix


BASE = np.array([[0.0, 2.0], [3.0, 1.0]])


def test_generation_is_reproducible_and_normalized() -> None:
    first = generate_flow_scenarios(BASE, scenario_count=5, seed=12)
    second = generate_flow_scenarios(BASE, scenario_count=5, seed=12)

    assert np.array_equal(first.flows, second.flows)
    assert np.all(first.flows >= 0)
    assert first.flows.shape == (5, 2, 2)
    assert np.allclose(first.flows.sum(axis=(1, 2)), 1.0)
    assert np.allclose(first.probabilities, 0.2)
    assert first.probabilities.sum() == pytest.approx(1.0)


def test_different_seeds_produce_different_scenarios() -> None:
    first = generate_flow_scenarios(BASE, scenario_count=5, seed=1)
    second = generate_flow_scenarios(BASE, scenario_count=5, seed=2)

    assert not np.array_equal(first.flows, second.flows)


def test_input_is_unchanged() -> None:
    original = BASE.copy()
    generate_flow_scenarios(original, scenario_count=3, seed=4)

    assert np.array_equal(original, BASE)


@pytest.mark.parametrize("scenario_count", [0, -1, 1.5, True])
def test_invalid_scenario_count_is_rejected(scenario_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_flow_scenarios(BASE, scenario_count=scenario_count)  # type: ignore[arg-type]


def test_zero_demand_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive total demand"):
        generate_flow_scenarios(np.zeros((2, 2)))


def test_repeated_zero_total_draws_raise_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ZeroDrawRng:
        uniform_calls = 0
        poisson_calls = 0

        def uniform(self, low: float, high: float, size: int) -> np.ndarray:
            self.uniform_calls += 1
            return np.ones(size)

        def poisson(self, rates: np.ndarray) -> np.ndarray:
            self.poisson_calls += 1
            return np.zeros_like(rates, dtype=int)

    rng = ZeroDrawRng()
    monkeypatch.setattr(scenario_module.np.random, "default_rng", lambda seed: rng)

    with pytest.raises(RuntimeError, match="zero total flow after 4 attempts"):
        generate_flow_scenarios(BASE, scenario_count=1, seed=3)

    assert rng.uniform_calls == 4
    assert rng.poisson_calls == 4


@pytest.mark.parametrize(
    ("filename", "size"),
    [
        ("wij_CAB25.xlsx", 25),
        ("wij_AP100.xlsx", 100),
        ("wij_AP150.xlsx", 150),
        ("wij_AP200.xlsx", 200),
    ],
)
def test_supplied_demand_dataset_smoke(filename: str, size: int) -> None:
    flow = load_matrix(Path("data/raw") / filename)
    scenarios = generate_flow_scenarios(flow, scenario_count=2, seed=99)

    assert scenarios.flows.shape == (2, size, size)
    assert np.allclose(scenarios.flows.sum(axis=(1, 2)), 1.0)
