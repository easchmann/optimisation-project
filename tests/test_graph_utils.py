"""Tests for graph_utils.py: generation, DSATUR, and brute force."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_utils import brute_force, brute_force_timed, dsatur, make_random_graph


class TestMakeRandomGraph:
    def test_node_count(self) -> None:
        G = make_random_graph(30, 0.5, seed=0)
        assert G.number_of_nodes() == 30

    def test_reproducible(self) -> None:
        G1 = make_random_graph(20, 0.4, seed=7)
        G2 = make_random_graph(20, 0.4, seed=7)
        assert set(G1.edges()) == set(G2.edges())

    def test_different_seeds_differ(self) -> None:
        G1 = make_random_graph(50, 0.5, seed=1)
        G2 = make_random_graph(50, 0.5, seed=2)
        # Extremely unlikely to be identical for n=50
        assert set(G1.edges()) != set(G2.edges())


class TestDSATUR:
    def test_valid_coloring_no_conflicts(self) -> None:
        G = make_random_graph(30, 0.5, seed=42)
        coloring = dsatur(G)
        for u, v in G.edges():
            assert coloring[u] != coloring[v], f"Conflict on edge ({u}, {v})"

    def test_all_vertices_coloured(self) -> None:
        G = make_random_graph(20, 0.4, seed=0)
        coloring = dsatur(G)
        assert set(coloring.keys()) == set(G.nodes())

    def test_triangle_needs_3_colours(self) -> None:
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        coloring = dsatur(G)
        assert len(set(coloring.values())) == 3

    def test_bipartite_needs_2_colours(self) -> None:
        G = nx.complete_bipartite_graph(3, 3)
        coloring = dsatur(G)
        assert len(set(coloring.values())) == 2

    def test_colours_are_1_indexed(self) -> None:
        G = make_random_graph(15, 0.4, seed=5)
        coloring = dsatur(G)
        assert min(coloring.values()) == 1


class TestBruteForce:
    def test_triangle_optimal(self) -> None:
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        coloring = brute_force(G, k_max=5)
        assert coloring is not None
        assert len(set(coloring.values())) == 3
        for u, v in G.edges():
            assert coloring[u] != coloring[v]

    def test_single_edge_needs_2(self) -> None:
        G = nx.Graph()
        G.add_edge(1, 2)
        coloring = brute_force(G, k_max=3)
        assert coloring is not None
        assert len(set(coloring.values())) == 2

    def test_empty_graph_needs_1(self) -> None:
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        coloring = brute_force(G, k_max=3)
        assert coloring is not None
        assert len(set(coloring.values())) == 1

    def test_returns_none_when_k_max_too_small(self) -> None:
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        assert brute_force(G, k_max=2) is None

    def test_timed_returns_elapsed(self) -> None:
        G = nx.Graph()
        G.add_edge(1, 2)
        coloring, elapsed = brute_force_timed(G, k_max=3)
        assert coloring is not None
        assert elapsed >= 0.0
