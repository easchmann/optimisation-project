"""Contract tests for all algorithm modules (ga, aco, sa)."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from algorithms import aco, ga, sa
from fitness import AlgoResult


def small_graph() -> nx.Graph:
    """Return a small Petersen-like graph suitable for fast algorithm tests."""
    return nx.petersen_graph()


def _assert_valid_result(result: AlgoResult, G: nx.Graph) -> None:
    """Assert that result satisfies the AlgoResult contract."""
    assert isinstance(result, AlgoResult)
    assert set(result.coloring.keys()) == set(G.nodes())
    assert all(c >= 1 for c in result.coloring.values())
    assert result.k_used == len(set(result.coloring.values()))
    assert result.violations >= 0
    assert result.runtime_s >= 0.0
    assert isinstance(result.fitness_history, list)


# ---------------------------------------------------------------------------
# GA / GAE
# ---------------------------------------------------------------------------

class TestGA:
    def test_returns_algo_result(self) -> None:
        G = small_graph()
        result = ga.run(G, k_max=10, params=dict(ga.DEFAULT_PARAMS), seed=0)
        _assert_valid_result(result, G)

    def test_feasible_solution(self) -> None:
        G = nx.cycle_graph(6)
        result = ga.run(G, k_max=6, params=dict(ga.DEFAULT_PARAMS), seed=1)
        assert result.violations == 0

    def test_gae_elitism_flag(self) -> None:
        G = small_graph()
        params = {**ga.DEFAULT_PARAMS, "elitism": True}
        result = ga.run(G, k_max=10, params=params, seed=0)
        _assert_valid_result(result, G)

    def test_deterministic_with_same_seed(self) -> None:
        G = small_graph()
        r1 = ga.run(G, k_max=10, params=dict(ga.DEFAULT_PARAMS), seed=42)
        r2 = ga.run(G, k_max=10, params=dict(ga.DEFAULT_PARAMS), seed=42)
        assert r1.coloring == r2.coloring

    def test_fitness_history_non_empty(self) -> None:
        G = small_graph()
        result = ga.run(G, k_max=10, params=dict(ga.DEFAULT_PARAMS), seed=0)
        assert len(result.fitness_history) > 0


# ---------------------------------------------------------------------------
# ACO
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="not yet implemented")
class TestACO:
    def test_returns_algo_result(self) -> None:
        G = small_graph()
        result = aco.run(G, k_max=10, params=dict(aco.DEFAULT_PARAMS), seed=0)
        _assert_valid_result(result, G)

    def test_feasible_solution(self) -> None:
        G = nx.cycle_graph(6)
        result = aco.run(G, k_max=6, params=dict(aco.DEFAULT_PARAMS), seed=1)
        assert result.violations == 0

    def test_deterministic_with_same_seed(self) -> None:
        G = small_graph()
        r1 = aco.run(G, k_max=10, params=dict(aco.DEFAULT_PARAMS), seed=7)
        r2 = aco.run(G, k_max=10, params=dict(aco.DEFAULT_PARAMS), seed=7)
        assert r1.coloring == r2.coloring

    def test_fitness_history_non_empty(self) -> None:
        G = small_graph()
        result = aco.run(G, k_max=10, params=dict(aco.DEFAULT_PARAMS), seed=0)
        assert len(result.fitness_history) > 0


# ---------------------------------------------------------------------------
# SA
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="not yet implemented")
class TestSA:
    def test_returns_algo_result(self) -> None:
        G = small_graph()
        result = sa.run(G, k_max=10, params=dict(sa.DEFAULT_PARAMS), seed=0)
        _assert_valid_result(result, G)

    def test_feasible_solution(self) -> None:
        G = nx.cycle_graph(6)
        result = sa.run(G, k_max=6, params=dict(sa.DEFAULT_PARAMS), seed=1)
        assert result.violations == 0

    def test_deterministic_with_same_seed(self) -> None:
        G = small_graph()
        r1 = sa.run(G, k_max=10, params=dict(sa.DEFAULT_PARAMS), seed=3)
        r2 = sa.run(G, k_max=10, params=dict(sa.DEFAULT_PARAMS), seed=3)
        assert r1.coloring == r2.coloring

    def test_n_step_defaults_to_5n(self) -> None:
        G = small_graph()
        params = {**sa.DEFAULT_PARAMS, "n_step": None}
        result = sa.run(G, k_max=10, params=params, seed=0)
        _assert_valid_result(result, G)
