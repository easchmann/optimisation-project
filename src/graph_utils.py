"""Graph generation, loading, DSATUR, brute force, and χ(G) lookup."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def make_random_graph(n: int, p: float, seed: int) -> nx.Graph:
    """Generate an Erdős–Rényi random graph.

    Args:
        n: Number of vertices.
        p: Edge probability.
        seed: Random seed for reproducibility.

    Returns:
        An undirected NetworkX graph.
    """
    rng = np.random.default_rng(seed)
    return nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**31)))


def load_dimacs(path: Path) -> nx.Graph:
    """Load a graph from a DIMACS .col file.

    Args:
        path: Path to the .col file.

    Returns:
        An undirected NetworkX graph with integer vertices.
    """
    G = nx.Graph()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("p"):
                _, _, n_str, _ = line.split()
                G.add_nodes_from(range(1, int(n_str) + 1))
            elif line.startswith("e"):
                _, u, v = line.split()
                G.add_edge(int(u), int(v))
    return G


def load_chromatic_numbers(path: Path) -> dict[str, int]:
    """Load known chromatic numbers from JSON.

    Args:
        path: Path to chromatic_numbers.json.

    Returns:
        Mapping from graph filename stem to χ(G).
    """
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# DSATUR baseline
# ---------------------------------------------------------------------------


def dsatur(G: nx.Graph) -> dict[int, int]:
    """Colour G with the DSATUR greedy heuristic.

    Selects the uncoloured vertex with the highest saturation degree
    (number of distinct colours in its neighbourhood); breaks ties by
    choosing the vertex with the highest degree.

    Args:
        G: An undirected NetworkX graph.

    Returns:
        Coloring dict mapping vertex -> colour (1-indexed).
    """
    coloring: dict[int, int] = {}
    saturation: dict[int, set[int]] = {v: set() for v in G.nodes()}

    for _ in range(G.number_of_nodes()):
        uncoloured = [v for v in G.nodes() if v not in coloring]
        # highest saturation, then highest degree as tiebreak
        vertex = max(uncoloured, key=lambda v: (len(saturation[v]), G.degree(v)))

        neighbour_colours = {coloring[u] for u in G.neighbors(vertex) if u in coloring}
        colour = 1
        while colour in neighbour_colours:
            colour += 1
        coloring[vertex] = colour

        for u in G.neighbors(vertex):
            if u not in coloring:
                saturation[u].add(colour)

    return coloring


# ---------------------------------------------------------------------------
# Brute-force baseline (only feasible for small graphs)
# ---------------------------------------------------------------------------


def brute_force(G: nx.Graph, k_max: int) -> dict[int, int] | None:
    """Find the optimal k-colouring by exhaustive search.

    Tries all k from 1 upward until a valid colouring is found or k_max
    is reached.  Intended only for very small graphs (n ≤ ~20).

    Args:
        G: An undirected NetworkX graph.
        k_max: Maximum number of colours to try.

    Returns:
        Optimal coloring dict, or None if no valid colouring found within k_max.
    """
    nodes = list(G.nodes())
    edges = list(G.edges())

    for k in range(1, k_max + 1):
        for assignment in product(range(1, k + 1), repeat=len(nodes)):
            coloring = dict(zip(nodes, assignment))
            if all(coloring[u] != coloring[v] for u, v in edges):
                return coloring
    return None


def brute_force_timed(
    G: nx.Graph, k_max: int
) -> tuple[dict[int, int] | None, float]:
    """Run brute_force and return (coloring, runtime_s).

    Args:
        G: An undirected NetworkX graph.
        k_max: Maximum number of colours to try.

    Returns:
        Tuple of (coloring or None, elapsed seconds).
    """
    t0 = time.perf_counter()
    result = brute_force(G, k_max)
    return result, time.perf_counter() - t0
