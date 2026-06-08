"""Simulated Annealing with restarts for graph colouring."""

from __future__ import annotations

import networkx as nx

from fitness import AlgoResult

DEFAULT_PARAMS: dict = {
    "T0": 100.0,
    "gamma": 0.995,
    # n_step is set to 5*n at runtime if not provided
    "n_step": None,
    "n_stall": 500,
}


def run(
    G: nx.Graph,
    k_max: int,
    params: dict,
    seed: int = 0,
) -> AlgoResult:
    """Run Simulated Annealing with restarts on the graph colouring problem.

    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Algorithm hyperparameters; falls back to DEFAULT_PARAMS for missing keys.
            n_step defaults to 5 * G.number_of_nodes() if None.
        seed: Random seed.

    Returns:
        AlgoResult with best coloring found.
    """
    raise NotImplementedError
