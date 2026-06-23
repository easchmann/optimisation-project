"""Hybrid GA + SA (Memetic Algorithm) for graph colouring.

Combines Genetic Algorithm's population-based global search with Simulated
Annealing's local optimization capabilities.
"""

from __future__ import annotations

import random
import time
from typing import Any

import networkx as nx
import numpy as np
from deap import base, creator, tools

from fitness import ALPHA, BETA, AlgoResult
from fitness import fitness as _fitness

# ── DEAP setup ──────────────────────────────────────────────────────────────────

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# ── SA refinement helpers ──────────────────────────────────────────────────────

def _sa_refine_solution(
    colours: np.ndarray,
    G: nx.Graph,
    nodes: list[int],
    adj: list[list[int]],
    k_max: int,
    n_steps: int,
    T0: float,
    gamma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    Run Simulated Annealing on a single solution for n_steps moves.
    This extracts the core SA logic from the original SA run() function.
    
    Returns:
        (improved_colours, final_fitness)
    """
    n = len(nodes)
    colours = colours.copy().astype(np.int32)
    
    # Compute initial state (same as SA run())
    colour_counts = np.bincount(colours, minlength=k_max + 1)[1:].astype(np.int32)
    k_used = int((colour_counts > 0).sum())
    violations = sum(
        1 for i in range(n) for j in adj[i] if j > i and colours[i] == colours[j]
    )
    current_f = float(ALPHA * violations + BETA * k_used)
    
    T = T0
    
    for _ in range(n_steps):
        # --- Propose move (copied from SA) ---
        i = int(rng.integers(0, n))
        old_c = int(colours[i])
        new_c = int(rng.integers(1, k_max))
        if new_c >= old_c:
            new_c += 1
        
        # Compute delta violations
        delta_v = 0
        for j in adj[i]:
            if colours[j] == old_c:
                delta_v -= 1
            if colours[j] == new_c:
                delta_v += 1
        
        # Update counts and compute delta fitness
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
            colour_counts[old_c - 1] += 1
            colour_counts[new_c - 1] -= 1
        
        # Cool
        T *= gamma
    
    return colours, current_f


def _numpy_to_deap(
    colours: np.ndarray,
    G: nx.Graph,
    nodes: list[int],
) -> creator.Individual:
    """Convert numpy coloring to DEAP Individual with fitness."""
    n = len(nodes)
    ind = creator.Individual(colours.tolist())
    coloring = {nodes[i]: int(colours[i]) for i in range(n)}
    ind.fitness.values = (_fitness(coloring, G),)
    return ind


def _deap_to_numpy(individual: list[int]) -> np.ndarray:
    """Convert DEAP Individual to numpy coloring."""
    return np.array(individual, dtype=np.int32)


def _build_hybrid_toolbox(
    G: nx.Graph,
    nodes: list[int],
    k_max: int,
    p_ind: float,
    t_size: int,
) -> base.Toolbox:
    """Build DEAP Toolbox (same as GA)."""
    n = len(nodes)
    idx_to_node = dict(enumerate(nodes))
    
    def evaluate(individual: list[int]) -> tuple[float]:
        coloring = {idx_to_node[i]: individual[i] for i in range(n)}
        return (_fitness(coloring, G),)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_color", random.randint, 1, k_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_color, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=k_max, indpb=p_ind)
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    return toolbox


# ── Main run function ──────────────────────────────────────────────────────────

def run(
    G: nx.Graph,
    k_max: int,
    params: dict[str, Any],
    seed: int = 0,
) -> AlgoResult:
    """
    Hybrid GA with SA local refinement (Memetic Algorithm).
    
    Combines GA's population-based global search with SA's local optimization.
    Every generation, the top individuals are refined using SA.
    
    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Algorithm hyperparameters. Missing keys fall back to defaults.
            GA parameters: n_pop, p_cx, p_mut, p_ind, n_gen, n_elite, t_size, elitism
            SA parameters: sa_steps (None → 50*n), sa_T0, sa_gamma, sa_refine_fraction
        seed: Random seed.
    
    Returns:
        AlgoResult with the best colouring found.
    """
    # ── Default parameters ────────────────────────────────────────────────────
    default_params = {
        # GA parameters (based on GAE defaults)
        "n_pop": 100,
        "p_cx": 0.8,
        "p_mut": 0.4,
        "p_ind": 0.02,
        "n_gen": 150,
        "n_elite": 3,
        "t_size": 5,
        "elitism": True,
        # SA refinement parameters
        "sa_steps": None,  # resolved to 50*n at runtime
        "sa_T0": 5.0,
        "sa_gamma": 0.95,
        "sa_refine_fraction": 0.15,
    }
    
    p = {**default_params, **params}
    
    n_pop: int = p["n_pop"]
    p_cx: float = p["p_cx"]
    p_mut: float = p["p_mut"]
    p_ind: float = p["p_ind"]
    n_gen: int = p["n_gen"]
    n_elite: int = p["n_elite"]
    t_size: int = p["t_size"]
    elitism: bool = p["elitism"]
    
    sa_steps: int = p["sa_steps"] if p["sa_steps"] is not None else 50 * len(G.nodes())
    sa_T0: float = p["sa_T0"]
    sa_gamma: float = p["sa_gamma"]
    sa_refine_fraction: float = p["sa_refine_fraction"]
    
    # ── Random seeds ──────────────────────────────────────────────────────────
    random.seed(seed)
    rng = np.random.default_rng(seed + 1)  # Separate RNG for SA
    
    # ── Setup ─────────────────────────────────────────────────────────────────
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx_to_node = dict(enumerate(nodes))
    
    # Precompute adjacency for SA
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    adj = [
        [node_to_idx[nb] for nb in G.neighbors(nodes[i])] for i in range(n)
    ]
    
    toolbox = _build_hybrid_toolbox(G, nodes, k_max, p_ind, t_size)
    
    t0 = time.perf_counter()
    
    # ── GA Initialization ─────────────────────────────────────────────────────
    population = toolbox.population(n=n_pop)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    fitness_history: list[float] = []
    
    # ── GA Evolution Loop ─────────────────────────────────────────────────────
    for gen in range(n_gen):
        # ── GA Selection and Variation ───────────────────────────────────────
        if elitism:
            elites = [toolbox.clone(e) for e in tools.selBest(population, n_elite)]
        
        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, n_pop)]
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_cx:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < p_mut:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        
        population[:] = offspring
        
        # Elitism
        if elitism:
            worst = sorted(range(n_pop), 
                          key=lambda i: population[i].fitness.values[0], 
                          reverse=True)
            for slot, elite in zip(worst[:n_elite], elites):
                population[slot] = elite
        
        # ── SA Refinement (The Hybrid Part) ─────────────────────────────────
        # Refine the top sa_refine_fraction of the population
        top_n = max(1, int(n_pop * sa_refine_fraction))
        sorted_indices = sorted(range(n_pop), 
                               key=lambda i: population[i].fitness.values[0])
        
        for idx in sorted_indices[:top_n]:
            # Convert to numpy, refine with SA
            colours_np = _deap_to_numpy(population[idx])
            improved_np, _ = _sa_refine_solution(
                colours_np, G, nodes, adj, k_max, 
                sa_steps, sa_T0, sa_gamma, rng
            )
            
            # Convert back to DEAP Individual
            improved_ind = _numpy_to_deap(improved_np, G, nodes)
            
            # Replace the original with the improved version
            population[idx] = improved_ind
        
        # ── Track best fitness ───────────────────────────────────────────────
        best_f = min(ind.fitness.values[0] for ind in population)
        fitness_history.append(best_f)
    
    runtime_s = time.perf_counter() - t0
    
    # ── Return best solution ─────────────────────────────────────────────────
    best = tools.selBest(population, 1)[0]
    coloring = {idx_to_node[i]: best[i] for i in range(n)}
    k_used = len(set(coloring.values()))
    violations = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])
    
    return AlgoResult(
        coloring=coloring,
        k_used=k_used,
        violations=violations,
        runtime_s=runtime_s,
        fitness_history=fitness_history,
    )


# ── Default parameters for config.py ──────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "n_pop": 100,
    "p_cx": 0.8,
    "p_mut": 0.4,
    "p_ind": 0.02,
    "n_gen": 150,
    "n_elite": 3,
    "t_size": 5,
    "elitism": True,
    "sa_steps": None,  # resolved to 50*n at runtime
    "sa_T0": 5.0,
    "sa_gamma": 0.95,
    "sa_refine_fraction": 0.15,
}