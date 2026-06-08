"""Simulated Annealing with restarts for graph colouring."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from fitness import ALPHA, BETA, AlgoResult

DEFAULT_PARAMS: dict = {
    "T0": 100.0,
    "gamma": 0.995,
    "n_step": None,   # resolved to 5*n at runtime
    "n_stall": 500,
    "n_max": None,    # resolved to 200*n_step at runtime
}


def _maybe_restart(
    colours: np.ndarray,
    best_colours: np.ndarray,
    stall_count: int,
    n_stall: int,
    T: float,
    T0: float,
) -> tuple[np.ndarray, float, int]:
    """Reset to best-so-far if the stall limit has been reached.

    Args:
        colours: Current colour assignment (not mutated).
        best_colours: Best-known colour assignment.
        stall_count: Consecutive moves without improvement in best F.
        n_stall: Threshold that triggers a restart.
        T: Current temperature.
        T0: Initial temperature; restart sets T = T0 * 0.1.

    Returns:
        (new_colours, new_T, new_stall_count) — new_colours is a fresh copy.
    """
    if stall_count >= n_stall:
        return best_colours.copy(), T0 * 0.1, 0
    return colours.copy(), T, stall_count


def run(
    G: nx.Graph,
    k_max: int,
    params: dict,
    seed: int = 0,
) -> AlgoResult:
    """Run Simulated Annealing with restarts on the graph colouring problem.

    Starts from a random colouring drawn uniformly from {1,...,k_max}^n.
    Each move reassigns one random vertex to a different colour, then applies
    Metropolis acceptance.  Temperature is multiplied by gamma every n_step
    moves.  If best F shows no improvement for n_stall consecutive moves the
    current solution is reset to best-so-far and T is dropped to T0*0.1.

    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Hyperparameters; missing keys fall back to DEFAULT_PARAMS.
            n_step defaults to 5*n, n_max to 200*n_step when None.
        seed: Random seed for the numpy Generator.

    Returns:
        AlgoResult with the best solution found across the entire run.
    """
    p = {**DEFAULT_PARAMS, **params}
    T0: float = p["T0"]
    gamma: float = p["gamma"]
    n_stall: int = p["n_stall"]

    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if k_max == 1:
        colours = {v: 1 for v in G.nodes()}
        violations = sum(1 for u, v in G.edges() if colours[u] == colours[v])
        return AlgoResult(coloring=colours, k_used=1, violations=violations,
                          runtime_s=0.0, fitness_history=[])

    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    n_step: int = p["n_step"] if p["n_step"] is not None else 5 * n
    n_max: int = p["n_max"] if p["n_max"] is not None else 200 * n_step

    # Adjacency as index lists for O(degree) delta computation.
    adj: list[list[int]] = [
        [node_to_idx[nb] for nb in G.neighbors(nodes[i])] for i in range(n)
    ]

    # Random initial colouring, fitness components.
    colours = rng.integers(1, k_max + 1, size=n).astype(np.int32)
    colour_counts = np.bincount(colours, minlength=k_max + 1)[1:].astype(np.int32)
    k_used = int((colour_counts > 0).sum())
    violations = sum(
        1 for i in range(n) for j in adj[i] if j > i and colours[i] == colours[j]
    )
    current_f = float(ALPHA * violations + BETA * k_used)

    best_colours = colours.copy()
    best_f = current_f
    best_violations = violations
    best_k_used = k_used

    T = T0
    stall_count = 0
    fitness_history: list[float] = []
    t_wall = time.perf_counter()

    for total_moves in range(1, n_max + 1):
        # --- propose move: one vertex, different colour ---
        i = int(rng.integers(0, n))
        old_c = int(colours[i])
        # Draw from {1,...,k_max-1} then shift to skip old_c.
        new_c = int(rng.integers(1, k_max))
        if new_c >= old_c:
            new_c += 1

        # Incremental delta_violations: only neighbours of i are affected.
        delta_v = 0
        for j in adj[i]:
            if colours[j] == old_c:
                delta_v -= 1
            if colours[j] == new_c:
                delta_v += 1

        # Incremental delta_k_used via colour_counts.
        colour_counts[old_c - 1] -= 1
        colour_counts[new_c - 1] += 1
        new_k_used = int((colour_counts > 0).sum())
        delta_k = new_k_used - k_used
        delta_f = float(ALPHA * delta_v + BETA * delta_k)

        # --- Metropolis acceptance ---
        accept = delta_f <= 0 or (T > 0.0 and rng.random() < np.exp(-delta_f / T))

        if accept:
            colours[i] = new_c
            violations += delta_v
            k_used = new_k_used
            current_f += delta_f
        else:
            colour_counts[old_c - 1] += 1   # revert counts
            colour_counts[new_c - 1] -= 1

        # --- track global best ---
        if current_f < best_f:
            best_f = current_f
            best_colours = colours.copy()
            best_violations = violations
            best_k_used = k_used
            stall_count = 0
        else:
            stall_count += 1

        # --- restart if stalled ---
        if stall_count >= n_stall:
            colours, T, stall_count = _maybe_restart(
                colours, best_colours, stall_count, n_stall, T, T0
            )
            violations = best_violations
            k_used = best_k_used
            colour_counts = np.bincount(colours, minlength=k_max + 1)[1:].astype(np.int32)
            current_f = best_f

        # --- cool and record every n_step moves ---
        if total_moves % n_step == 0:
            T *= gamma
            fitness_history.append(best_f)

    runtime_s = time.perf_counter() - t_wall

    coloring = {nodes[i]: int(best_colours[i]) for i in range(n)}
    k_final = len(set(coloring.values()))
    v_final = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])

    return AlgoResult(
        coloring=coloring,
        k_used=k_final,
        violations=v_final,
        runtime_s=runtime_s,
        fitness_history=fitness_history,
    )
