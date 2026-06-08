"""Fitness function and result dataclass for graph coloring."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

ALPHA = 1000  # violation penalty weight (hard constraint)
BETA = 1      # colour count weight (soft objective)


@dataclass
class AlgoResult:
    """Uniform return type for all algorithm modules."""

    coloring: dict[int, int]        # vertex -> colour (1-indexed)
    k_used: int                     # number of distinct colours used
    violations: int                 # number of conflicting edges
    runtime_s: float                # wall-clock seconds
    fitness_history: list[float] = field(default_factory=list)  # best F per iteration


def fitness(coloring: dict[int, int], G: nx.Graph) -> float:
    """Compute weighted fitness: ALPHA * violations + BETA * colours used.

    Args:
        coloring: Mapping from vertex to colour (1-indexed).
        G: The graph being coloured.

    Returns:
        Scalar fitness value; lower is better.
    """
    violations = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])
    k_used = len(set(coloring.values()))
    return ALPHA * violations + BETA * k_used


def chromatic_gap(k_used: int, chi: int) -> int:
    """Compute gap between colours used and a reference colouring count.

    Args:
        k_used: Number of colours used by the algorithm.
        chi: Reference colour count — the true χ(G) or a heuristic result
            (e.g. DSATUR).  When chi is a heuristic, the gap may be negative
            if the algorithm finds a better colouring.

    Returns:
        Integer gap (k_used - chi); 0 means equal to reference, negative means
        better than reference.
    """
    return k_used - chi
