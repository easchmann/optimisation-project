"""Tests for fitness.py: fitness function and AlgoResult dataclass."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fitness import ALPHA, BETA, AlgoResult, chromatic_gap, fitness


def triangle() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    return G


class TestFitness:
    def test_feasible_3coloring_triangle(self) -> None:
        G = triangle()
        coloring = {1: 1, 2: 2, 3: 3}
        assert fitness(coloring, G) == BETA * 3

    def test_infeasible_2coloring_triangle(self) -> None:
        # Assigning same colour to vertices 1 and 3 creates a violation
        G = triangle()
        coloring = {1: 1, 2: 2, 3: 1}
        f = fitness(coloring, G)
        assert f == ALPHA * 1 + BETA * 2

    def test_feasible_beats_infeasible(self) -> None:
        """A valid colouring with extra colour must score below any infeasible solution."""
        G = triangle()
        feasible = {1: 1, 2: 2, 3: 3}    # 3 colours, 0 violations
        infeasible = {1: 1, 2: 1, 3: 1}  # 1 colour, 3 violations
        assert fitness(feasible, G) < fitness(infeasible, G)

    def test_single_vertex(self) -> None:
        G = nx.Graph()
        G.add_node(1)
        assert fitness({1: 1}, G) == BETA * 1

    def test_empty_graph(self) -> None:
        G = nx.Graph()
        G.add_nodes_from([1, 2])
        coloring = {1: 1, 2: 2}
        assert fitness(coloring, G) == BETA * 2


class TestChromaticGap:
    def test_optimal(self) -> None:
        assert chromatic_gap(3, 3) == 0

    def test_suboptimal(self) -> None:
        assert chromatic_gap(5, 3) == 2

    def test_gap_positive_when_suboptimal(self) -> None:
        assert chromatic_gap(4, 3) == 1

    def test_gap_negative_when_beats_reference(self) -> None:
        # Algorithm finds a better colouring than a heuristic reference
        assert chromatic_gap(2, 3) == -1


class TestAlgoResult:
    def test_instantiation(self) -> None:
        result = AlgoResult(
            coloring={1: 1, 2: 2},
            k_used=2,
            violations=0,
            runtime_s=0.01,
            fitness_history=[10.0, 5.0, 2.0],
        )
        assert result.k_used == 2
        assert result.violations == 0

    def test_default_fitness_history(self) -> None:
        result = AlgoResult(coloring={1: 1}, k_used=1, violations=0, runtime_s=0.0)
        assert result.fitness_history == []
