"""ACO + SA Hybrid for graph colouring.

Uses ACO to find high-quality solutions, then refines the best with SA.
"""

from __future__ import annotations

import time
from typing import Any

import networkx as nx
import numpy as np

from algorithms import aco
from algorithms.hybrid_ga_sa import _sa_refine_solution  # Reuse SA refinement
from fitness import ALPHA, BETA, AlgoResult


def run(
    G: nx.Graph,
    k_max: int,
    params: dict[str, Any],
    seed: int = 0,
) -> AlgoResult:
    """
    ACO + SA Hybrid: ACO finds good solutions, SA refines the best.
    
    Args:
        G: The graph to colour.
        k_max: Maximum number of colours.
        params: Hyperparameters.
        seed: Random seed.
    
    Returns:
        AlgoResult with best colouring.
    """
    # ── Parameters ──────────────────────────────────────────────────────────────
    default_params = {
        # ACO phase
        "aco_iter": 100,
        "aco_ants": 50,
        "aco_alpha": 2.0,
        "aco_beta": 5.0,
        "aco_rho": 0.2,
        "aco_Q": 1.0,
        
        # SA refinement phase
        "sa_steps": None,      # Will become sa_steps_factor*n at runtime
        "sa_steps_factor": 100,
        "sa_T0": 20.0,         # Higher temp for more exploration
        "sa_gamma": 0.99,
        "sa_restarts": 3,      # Number of SA restarts
    }

    p = {**default_params, **params}

    aco_iter: int = p["aco_iter"]
    aco_ants: int = p["aco_ants"]
    aco_alpha: float = p["aco_alpha"]
    aco_beta: float = p["aco_beta"]
    aco_rho: float = p["aco_rho"]
    aco_Q: float = p["aco_Q"]

    sa_steps: int = p["sa_steps"] if p["sa_steps"] is not None else p["sa_steps_factor"] * len(G.nodes())
    sa_T0: float = p["sa_T0"]
    sa_gamma: float = p["sa_gamma"]
    sa_restarts: int = p["sa_restarts"]
    
    rng = np.random.default_rng(seed)
    
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx_to_node = {v: i for i, v in enumerate(nodes)}
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Precompute adjacency for SA
    adj = [
        [node_to_idx[nb] for nb in G.neighbors(nodes[i])] for i in range(n)
    ]
    
    t0 = time.perf_counter()

    # ── Phase 1: Run ACO ──────────────────────────────────────────────────────
    aco_params = {
        "n_ants": aco_ants,
        "alpha": aco_alpha,
        "beta": aco_beta,
        "rho": aco_rho,
        "Q": aco_Q,
        "tau_min": 0.01,
        "n_iter": aco_iter,
    }

    result_aco = aco.run(G, k_max, aco_params, seed=seed)
    best_colours = np.array([result_aco.coloring[v] for v in nodes], dtype=np.int32)
    best_f = float(ALPHA * result_aco.violations + BETA * result_aco.k_used)
    fitness_history: list[float] = [best_f]

    # ── Phase 2: SA Refinement with Restarts ──────────────────────────────────
    for restart in range(sa_restarts):
        # Run SA refinement with different random seeds
        refined, final_f = _sa_refine_solution(
            best_colours, G, nodes, adj, k_max,
            sa_steps, sa_T0, sa_gamma,
            rng
        )

        # Compare on full fitness (violations + colour count), not colour count
        # alone — a refinement that trades a conflict for one fewer colour must
        # never be preferred, per ALPHA >> BETA.
        if final_f < best_f:
            best_f = final_f
            best_colours = refined.copy()

        fitness_history.append(best_f)

        # Cool down for next restart
        sa_T0 *= 0.9  # Slightly lower starting temp each restart

    runtime_s = time.perf_counter() - t0

    # ── Return best solution ──────────────────────────────────────────────────
    coloring = {nodes[i]: int(best_colours[i]) for i in range(n)}
    k_used = len(set(coloring.values()))
    violations = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])

    return AlgoResult(
        coloring=coloring,
        k_used=k_used,
        violations=violations,
        runtime_s=runtime_s,
        fitness_history=fitness_history,
    )


DEFAULT_PARAMS: dict = {
    "aco_iter": 100,
    "aco_ants": 50,
    "aco_alpha": 2.0,
    "aco_beta": 5.0,
    "aco_rho": 0.2,
    "aco_Q": 1.0,
    "sa_steps": None,      # resolved to sa_steps_factor*n at runtime
    "sa_steps_factor": 100,
    "sa_T0": 20.0,
    "sa_gamma": 0.99,
    "sa_restarts": 3,
}