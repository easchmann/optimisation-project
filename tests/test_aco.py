"""Dedicated tests for algorithms/aco.py."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from algorithms import aco
from algorithms.aco import _construct, _update_pheromone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_params(**overrides) -> dict:
    """Return DEFAULT_PARAMS with fast settings for unit tests."""
    return {**aco.DEFAULT_PARAMS, "n_ants": 10, "n_iter": 50, **overrides}


# ---------------------------------------------------------------------------
# Correctness: P4 path graph
# ---------------------------------------------------------------------------

class TestP4Colouring:
    """P4 = path 0-1-2-3.  Bipartite, χ=2."""

    def test_finds_valid_colouring(self) -> None:
        G = nx.path_graph(4)
        result = aco.run(G, k_max=4, params=_fast_params(n_iter=150), seed=0)
        assert result.violations == 0

    def test_uses_at_most_2_colours(self) -> None:
        G = nx.path_graph(4)
        result = aco.run(G, k_max=4, params=_fast_params(n_iter=200, n_ants=20), seed=1)
        assert result.k_used <= 2

    def test_colours_are_1_indexed(self) -> None:
        G = nx.path_graph(4)
        result = aco.run(G, k_max=4, params=_fast_params(), seed=0)
        assert all(1 <= c <= 4 for c in result.coloring.values())

    def test_all_vertices_coloured(self) -> None:
        G = nx.path_graph(4)
        result = aco.run(G, k_max=4, params=_fast_params(), seed=0)
        assert set(result.coloring.keys()) == set(G.nodes())


# ---------------------------------------------------------------------------
# Pheromone matrix invariants (via _update_pheromone)
# ---------------------------------------------------------------------------

class TestPheromoneMatrix:
    def test_shape_preserved(self) -> None:
        """tau must stay (n, k_max) after an update."""
        n, k_max = 5, 4
        tau = np.ones((n, k_max))
        colours = np.array([1, 2, 1, 3, 2], dtype=np.int32)
        _update_pheromone(tau, colours, f_best=5.0, rho=0.2, Q=1.0, tau_min=0.01)
        assert tau.shape == (n, k_max)

    def test_no_value_below_tau_min(self) -> None:
        """After update every cell must be >= tau_min."""
        n, k_max = 6, 3
        tau_min = 0.05
        tau = np.ones((n, k_max)) * 0.1  # start close to tau_min
        colours = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
        # Many evaporation steps to drive values low.
        for _ in range(50):
            _update_pheromone(tau, colours, f_best=10.0, rho=0.5, Q=0.01, tau_min=tau_min)
        assert np.all(tau >= tau_min - 1e-12)

    def test_deposit_increases_used_colour(self) -> None:
        """The pheromone cell for (vertex, used_colour) must increase after deposit."""
        n, k_max = 3, 2
        rho = 0.0   # no evaporation so increase is clean
        tau = np.ones((n, k_max))
        tau_before = tau.copy()
        colours = np.array([1, 2, 1], dtype=np.int32)
        _update_pheromone(tau, colours, f_best=2.0, rho=rho, Q=1.0, tau_min=0.0)
        # vertex 0 used colour 1 -> column 0
        assert tau[0, 0] > tau_before[0, 0]
        # vertex 1 used colour 2 -> column 1
        assert tau[1, 1] > tau_before[1, 1]

    def test_evaporation_reduces_unused_colour(self) -> None:
        """Cells for colours not used by the best ant must decrease (evaporation)."""
        n, k_max = 2, 2
        tau = np.ones((n, k_max)) * 2.0
        colours = np.array([1, 1], dtype=np.int32)  # only colour 1 used
        _update_pheromone(tau, colours, f_best=1.0, rho=0.5, Q=0.0, tau_min=0.0)
        # Column 1 (colour 2) was never deposited on; evaporation reduces it.
        assert tau[0, 1] < 2.0
        assert tau[1, 1] < 2.0


# ---------------------------------------------------------------------------
# Pheromone shape during a full run (via coloring bounds)
# ---------------------------------------------------------------------------

class TestColourBounds:
    def test_colours_within_k_max(self) -> None:
        """All assigned colours must be in 1..k_max regardless of k_max value."""
        for k_max in [3, 5, 8]:
            G = nx.cycle_graph(6)
            result = aco.run(G, k_max=k_max, params=_fast_params(), seed=0)
            assert all(1 <= c <= k_max for c in result.coloring.values()), (
                f"k_max={k_max}: colour out of range"
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_coloring(self) -> None:
        G = nx.petersen_graph()
        r1 = aco.run(G, k_max=10, params=_fast_params(), seed=99)
        r2 = aco.run(G, k_max=10, params=_fast_params(), seed=99)
        assert r1.coloring == r2.coloring

    def test_different_seeds_may_differ(self) -> None:
        G = nx.petersen_graph()
        r1 = aco.run(G, k_max=10, params=_fast_params(), seed=0)
        r2 = aco.run(G, k_max=10, params=_fast_params(), seed=1)
        # Not guaranteed to differ, but extremely likely on a non-trivial graph.
        assert r1.fitness_history != r2.fitness_history or r1.coloring != r2.coloring


# ---------------------------------------------------------------------------
# AlgoResult contract
# ---------------------------------------------------------------------------

class TestAlgoResultContract:
    def test_k_used_matches_coloring(self) -> None:
        G = nx.cycle_graph(6)
        result = aco.run(G, k_max=6, params=_fast_params(), seed=0)
        assert result.k_used == len(set(result.coloring.values()))

    def test_violations_matches_coloring(self) -> None:
        G = nx.cycle_graph(6)
        result = aco.run(G, k_max=6, params=_fast_params(), seed=0)
        expected = sum(1 for u, v in G.edges() if result.coloring[u] == result.coloring[v])
        assert result.violations == expected

    def test_fitness_history_length(self) -> None:
        n_iter = 30
        G = nx.path_graph(4)
        result = aco.run(G, k_max=4, params=_fast_params(n_iter=n_iter), seed=0)
        assert len(result.fitness_history) == n_iter

    def test_fitness_history_non_increasing(self) -> None:
        """Global best must never worsen across iterations."""
        G = nx.petersen_graph()
        result = aco.run(G, k_max=10, params=_fast_params(n_iter=50), seed=0)
        h = result.fitness_history
        assert all(h[i] >= h[i + 1] or h[i] == h[i + 1] for i in range(len(h) - 1))

    def test_runtime_positive(self) -> None:
        G = nx.path_graph(4)
        result = aco.run(G, k_max=4, params=_fast_params(), seed=0)
        assert result.runtime_s > 0.0
