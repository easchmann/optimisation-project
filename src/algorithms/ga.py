"""Genetic Algorithm (GA) and GA with Elitism (GAE) for graph colouring."""

from __future__ import annotations

import time

import networkx as nx

from fitness import AlgoResult, fitness

DEFAULT_PARAMS: dict = {
    "n_pop": 100,
    "p_cx": 0.5,
    "p_mut": 0.2,
    "p_ind": 0.05,
    "n_gen": 200,
    "n_elite": 3,   # GAE only
    "t_size": 4,
    "elitism": False,
}


def run(
    G: nx.Graph,
    k_max: int,
    params: dict,
    seed: int = 0,
) -> AlgoResult:
    """Run GA (or GAE when params['elitism'] is True) on the graph colouring problem.

    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Algorithm hyperparameters; falls back to DEFAULT_PARAMS for missing keys.
        seed: Random seed.

    Returns:
        AlgoResult with best coloring found.
    """
    raise NotImplementedError
