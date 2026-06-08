"""Ant Colony Optimisation for graph colouring."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from fitness import AlgoResult
from fitness import fitness as _fitness

DEFAULT_PARAMS: dict = {
    "n_ants": 50,
    "alpha": 1.0,
    "beta": 3.0,
    "rho": 0.2,
    "Q": 1.0,
    "tau_min": 0.01,
    "n_iter": 300,
}


def _construct(
    nodes: list[int],
    nb_idx: dict[int, list[int]],  # vertex -> list of neighbour positions in nodes
    tau: np.ndarray,
    alpha: float,
    beta: float,
    k_max: int,
    order: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one ant's complete colouring.

    Args:
        nodes: Sorted vertex list; index i <-> nodes[i].
        nb_idx: For each vertex, the sorted-list positions of its neighbours.
        tau: Pheromone matrix of shape (n, k_max).
        alpha: Pheromone exponent.
        beta: Heuristic exponent.
        k_max: Number of available colours (colours are 1-indexed).
        order: Permutation of range(n) giving the vertex visit sequence.
        rng: Seeded numpy Generator.

    Returns:
        Integer array of shape (n,) with colour assignments in 1..k_max.
    """
    n = len(nodes)
    colours = np.zeros(n, dtype=np.int32)

    for pos in order:
        # Count how many already-coloured neighbours carry each colour.
        conflict_count = np.zeros(k_max, dtype=np.float64)
        for nb_pos in nb_idx[nodes[pos]]:
            c = colours[nb_pos]
            if c > 0:
                conflict_count[c - 1] += 1.0

        eta = 1.0 / (1.0 + conflict_count)           # shape (k_max,)
        weights = (tau[pos] ** alpha) * (eta ** beta)
        total = weights.sum()
        probs = weights / total if total > 0.0 else np.ones(k_max) / k_max
        colours[pos] = int(rng.choice(k_max, p=probs)) + 1  # 1-indexed

    return colours


def _update_pheromone(
    tau: np.ndarray,
    best_colours: np.ndarray,
    f_best: float,
    rho: float,
    Q: float,
    tau_min: float,
) -> np.ndarray:
    """Evaporate, deposit from the best ant, and clip to tau_min.

    Args:
        tau: Current pheromone matrix, shape (n, k_max). Modified in-place.
        best_colours: Colour assignments from the best ant (1-indexed, length n).
        f_best: Fitness of the best ant this iteration.
        rho: Evaporation rate.
        Q: Deposit numerator constant.
        tau_min: Lower bound on pheromone values.

    Returns:
        Updated pheromone matrix (same object, also returned for convenience).
    """
    tau *= 1.0 - rho
    deposit = Q / max(f_best, 1e-9)
    for pos, c in enumerate(best_colours):
        tau[pos, c - 1] += deposit
    np.clip(tau, tau_min, None, out=tau)
    return tau


def run(
    G: nx.Graph,
    k_max: int,
    params: dict,
    seed: int = 0,
) -> AlgoResult:
    """Run Ant Colony Optimisation on the graph colouring problem.

    Each iteration all ants construct a full colouring, each visiting vertices
    in its own independently-shuffled order.  Pheromone is evaporated globally
    then deposited only by the best ant of that iteration.

    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Algorithm hyperparameters; missing keys fall back to DEFAULT_PARAMS.
        seed: Random seed for the numpy Generator.

    Returns:
        AlgoResult with best coloring found across all iterations.
    """
    p = {**DEFAULT_PARAMS, **params}
    n_ants: int = p["n_ants"]
    alpha: float = p["alpha"]
    beta: float = p["beta"]
    rho: float = p["rho"]
    Q: float = p["Q"]
    tau_min: float = p["tau_min"]
    n_iter: int = p["n_iter"]

    rng = np.random.default_rng(seed)

    nodes = sorted(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Precompute neighbour positions for fast conflict counting.
    nb_idx: dict[int, list[int]] = {
        v: [node_to_idx[nb] for nb in G.neighbors(v)]
        for v in nodes
    }

    tau = np.ones((n, k_max), dtype=np.float64)

    best_colours_global: np.ndarray | None = None
    best_f_global = float("inf")
    fitness_history: list[float] = []

    t0 = time.perf_counter()

    for _ in range(n_iter):
        iter_best_colours: np.ndarray | None = None
        iter_best_f = float("inf")

        for _ in range(n_ants):
            order = rng.permutation(n)  # each ant explores its own vertex order
            colours = _construct(nodes, nb_idx, tau, alpha, beta, k_max, order, rng)
            coloring = {nodes[i]: int(colours[i]) for i in range(n)}
            f = _fitness(coloring, G)
            if f < iter_best_f:
                iter_best_f = f
                iter_best_colours = colours.copy()

        _update_pheromone(tau, iter_best_colours, iter_best_f, rho, Q, tau_min)  # type: ignore[arg-type]

        if iter_best_f < best_f_global:
            best_f_global = iter_best_f
            best_colours_global = iter_best_colours.copy()  # type: ignore[union-attr]

        fitness_history.append(best_f_global)

    runtime_s = time.perf_counter() - t0

    if best_colours_global is None:
        raise RuntimeError("ACO produced no solution — n_iter must be >= 1")
    coloring = {nodes[i]: int(best_colours_global[i]) for i in range(n)}
    k_used = len(set(coloring.values()))
    violations = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])

    return AlgoResult(
        coloring=coloring,
        k_used=k_used,
        violations=violations,
        runtime_s=runtime_s,
        fitness_history=fitness_history,
    )
