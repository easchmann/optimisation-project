"""Ant Colony Optimisation for graph colouring."""

from __future__ import annotations

import networkx as nx

from fitness import AlgoResult

DEFAULT_PARAMS: dict = {
    "n_ants": 50,
    "alpha": 1.0,
    "beta": 3.0,
    "rho": 0.2,
    "n_iter": 300,
}


def run(
    G: nx.Graph,
    k_max: int,
    params: dict,
    seed: int = 0,
) -> AlgoResult:
    """Run Ant Colony Optimisation on the graph colouring problem.

    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Algorithm hyperparameters; falls back to DEFAULT_PARAMS for missing keys.
        seed: Random seed.

    Returns:
        AlgoResult with best coloring found.
    """
    raise NotImplementedError
