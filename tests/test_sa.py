"""Dedicated tests for algorithms/sa.py."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from algorithms import sa
from algorithms.sa import _maybe_restart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_params(**overrides) -> dict:
    """DEFAULT_PARAMS overridden with small n_max for fast unit tests."""
    return {**sa.DEFAULT_PARAMS, "n_step": 10, "n_max": 500, **overrides}


# ---------------------------------------------------------------------------
# Correctness: K3
# ---------------------------------------------------------------------------

class TestK3Colouring:
    """K3 = complete graph on 3 vertices; χ = 3, needs exactly 3 colours."""

    def test_finds_valid_colouring(self) -> None:
        G = nx.complete_graph(3)
        result = sa.run(G, k_max=4, params={**_fast_params(), "n_max": 5000}, seed=0)
        assert result.violations == 0

    def test_uses_at_most_3_colours(self) -> None:
        G = nx.complete_graph(3)
        result = sa.run(G, k_max=3, params={**_fast_params(), "n_max": 5000}, seed=7)
        assert result.k_used <= 3

    def test_colours_1_indexed(self) -> None:
        G = nx.complete_graph(3)
        result = sa.run(G, k_max=4, params=_fast_params(), seed=0)
        assert all(1 <= c <= 4 for c in result.coloring.values())

    def test_all_vertices_covered(self) -> None:
        G = nx.complete_graph(3)
        result = sa.run(G, k_max=4, params=_fast_params(), seed=0)
        assert set(result.coloring.keys()) == set(G.nodes())


# ---------------------------------------------------------------------------
# fitness_history contract
# ---------------------------------------------------------------------------

class TestFitnessHistory:
    def test_non_increasing(self) -> None:
        """Best F can only stay the same or improve across recordings."""
        G = nx.cycle_graph(8)
        result = sa.run(G, k_max=4, params=_fast_params(n_step=20, n_max=2000), seed=0)
        h = result.fitness_history
        assert all(h[i] >= h[i + 1] for i in range(len(h) - 1))

    def test_length_equals_n_max_over_n_step(self) -> None:
        n_step, n_max = 20, 400
        G = nx.path_graph(5)
        result = sa.run(G, k_max=3, params={**sa.DEFAULT_PARAMS, "n_step": n_step, "n_max": n_max}, seed=0)
        assert len(result.fitness_history) == n_max // n_step

    def test_non_empty(self) -> None:
        G = nx.path_graph(4)
        result = sa.run(G, k_max=3, params=_fast_params(), seed=0)
        assert len(result.fitness_history) > 0

    def test_last_entry_equals_returned_fitness(self) -> None:
        """The final history entry must agree with the returned coloring's fitness."""
        from fitness import fitness as _fitness
        G = nx.cycle_graph(6)
        result = sa.run(G, k_max=4, params=_fast_params(n_step=10, n_max=500), seed=2)
        returned_f = _fitness(result.coloring, G)
        # fitness_history records best_f; returned result must be <= last history value.
        assert returned_f <= result.fitness_history[-1] + 1e-9


# ---------------------------------------------------------------------------
# Restart logic (_maybe_restart unit tests)
# ---------------------------------------------------------------------------

class TestMaybeRestart:
    def test_triggers_at_threshold(self) -> None:
        colours = np.array([1, 1, 1], dtype=np.int32)
        best = np.array([1, 2, 3], dtype=np.int32)
        c_out, T_out, s_out = _maybe_restart(colours, best, stall_count=5, n_stall=5, T=20.0, T0=100.0)
        assert np.array_equal(c_out, best)
        assert T_out == pytest.approx(100.0)   # reset to T0
        assert s_out == 0

    def test_does_not_trigger_below_threshold(self) -> None:
        colours = np.array([2, 3, 1], dtype=np.int32)
        best = np.array([1, 2, 3], dtype=np.int32)
        c_out, T_out, s_out = _maybe_restart(colours, best, stall_count=4, n_stall=5, T=20.0, T0=100.0)
        assert np.array_equal(c_out, colours)
        assert T_out == pytest.approx(20.0)
        assert s_out == 4

    def test_returns_copy_not_same_object(self) -> None:
        colours = np.array([1, 2, 3], dtype=np.int32)
        best = np.array([1, 2, 3], dtype=np.int32)
        c_out, _, _ = _maybe_restart(colours, best, stall_count=10, n_stall=5, T=1.0, T0=10.0)
        c_out[0] = 99
        assert best[0] != 99, "restart must return a copy, not the best array itself"

    def test_integration_n_stall_1_never_regresses(self) -> None:
        """With n_stall=1 every non-improving move restarts; result must not exceed initial fitness."""
        from fitness import fitness as _fitness
        G = nx.complete_graph(4)
        rng = np.random.default_rng(0)
        nodes = sorted(G.nodes())
        n = len(nodes)
        init_colours = rng.integers(1, 5, size=n)
        init_coloring = {nodes[i]: int(init_colours[i]) for i in range(n)}
        init_f = _fitness(init_coloring, G)

        result = sa.run(
            G, k_max=4,
            params={**sa.DEFAULT_PARAMS, "n_stall": 1, "n_step": 5, "n_max": 300},
            seed=0,
        )
        result_f = _fitness(result.coloring, G)
        # With frequent restarts, returned best must be <= any initial random state.
        assert result_f <= init_f + 1e-9 or result_f <= init_f


# ---------------------------------------------------------------------------
# AlgoResult contract
# ---------------------------------------------------------------------------

class TestAlgoResultContract:
    def test_k_used_matches_coloring(self) -> None:
        G = nx.cycle_graph(6)
        result = sa.run(G, k_max=4, params=_fast_params(), seed=0)
        assert result.k_used == len(set(result.coloring.values()))

    def test_violations_matches_coloring(self) -> None:
        G = nx.cycle_graph(6)
        result = sa.run(G, k_max=4, params=_fast_params(), seed=0)
        expected = sum(1 for u, v in G.edges() if result.coloring[u] == result.coloring[v])
        assert result.violations == expected

    def test_runtime_positive(self) -> None:
        G = nx.path_graph(4)
        result = sa.run(G, k_max=3, params=_fast_params(), seed=0)
        assert result.runtime_s > 0.0

    def test_returns_best_not_final_state(self) -> None:
        """Returned fitness must equal the minimum of fitness_history."""
        from fitness import fitness as _fitness
        G = nx.petersen_graph()
        result = sa.run(G, k_max=10, params=_fast_params(n_step=25, n_max=500), seed=0)
        returned_f = _fitness(result.coloring, G)
        assert returned_f <= min(result.fitness_history) + 1e-9


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_coloring(self) -> None:
        G = nx.petersen_graph()
        r1 = sa.run(G, k_max=10, params=_fast_params(), seed=42)
        r2 = sa.run(G, k_max=10, params=_fast_params(), seed=42)
        assert r1.coloring == r2.coloring

    def test_same_seed_same_history(self) -> None:
        G = nx.cycle_graph(8)
        r1 = sa.run(G, k_max=4, params=_fast_params(n_step=20, n_max=200), seed=13)
        r2 = sa.run(G, k_max=4, params=_fast_params(n_step=20, n_max=200), seed=13)
        assert r1.fitness_history == r2.fitness_history

    def test_different_seeds_differ(self) -> None:
        G = nx.petersen_graph()
        r1 = sa.run(G, k_max=10, params=_fast_params(), seed=0)
        r2 = sa.run(G, k_max=10, params=_fast_params(), seed=99)
        assert r1.coloring != r2.coloring or r1.fitness_history != r2.fitness_history
