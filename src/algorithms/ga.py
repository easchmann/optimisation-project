"""Genetic Algorithm (GA) and GA with Elitism (GAE) for graph colouring."""

from __future__ import annotations

import random
import time

import networkx as nx
from deap import base, creator, tools

from fitness import AlgoResult
from fitness import fitness as _fitness

DEFAULT_PARAMS: dict = {
    "n_pop": 100,
    "p_cx": 0.5,
    "p_mut": 0.2,
    "p_ind": 0.05,
    "n_gen": 200,
    "n_elite": 3,  # GAE only
    "t_size": 4,
    "elitism": False,
}

# DEAP creator classes live in a global registry; guard against double-registration.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def _build_toolbox(
    G: nx.Graph,
    nodes: list[int],
    k_max: int,
    p_ind: float,
    t_size: int,
) -> base.Toolbox:
    """Construct a DEAP Toolbox wired to G and the given hyperparameters.

    Args:
        G: The graph being coloured.
        nodes: Sorted list of vertex IDs (defines the individual's index mapping).
        k_max: Maximum colour index (colours are 1..k_max inclusive).
        p_ind: Per-gene mutation probability.
        t_size: Tournament selection size.

    Returns:
        Configured DEAP Toolbox.
    """
    n = len(nodes)
    idx_to_node = dict(enumerate(nodes))

    def evaluate(individual: list[int]) -> tuple[float]:
        coloring = {idx_to_node[i]: individual[i] for i in range(n)}
        return (_fitness(coloring, G),)

    toolbox = base.Toolbox()
    toolbox.register("attr_color", random.randint, 1, k_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_color, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=k_max, indpb=p_ind)
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    return toolbox


def run(
    G: nx.Graph,
    k_max: int,
    params: dict,
    seed: int = 0,
) -> AlgoResult:
    """Run GA (or GAE when params['elitism'] is True) on the graph colouring problem.

    Uses two-point crossover, per-gene uniform-int mutation, and tournament
    selection. GAE re-injects the top n_elite individuals after each generation
    by overwriting the worst slots, guaranteeing no fitness regression.

    Args:
        G: The graph to colour.
        k_max: Maximum number of colours (search space upper bound).
        params: Algorithm hyperparameters; missing keys fall back to DEFAULT_PARAMS.
        seed: Random seed (controls all stochastic operators via Python's random).

    Returns:
        AlgoResult with best coloring found, per-generation fitness history.
    """
    p = {**DEFAULT_PARAMS, **params}
    n_pop: int = p["n_pop"]
    p_cx: float = p["p_cx"]
    p_mut: float = p["p_mut"]
    p_ind: float = p["p_ind"]
    n_gen: int = p["n_gen"]
    n_elite: int = p["n_elite"]
    t_size: int = p["t_size"]
    elitism: bool = p["elitism"]

    # DEAP's operators (cxTwoPoint, mutUniformInt, selTournament) use Python's
    # random module internally, so we seed it here for full reproducibility.
    random.seed(seed)

    nodes = sorted(G.nodes())
    n = len(nodes)
    idx_to_node = dict(enumerate(nodes))
    toolbox = _build_toolbox(G, nodes, k_max, p_ind, t_size)

    t0 = time.perf_counter()

    population = toolbox.population(n=n_pop)
    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit

    fitness_history: list[float] = []

    for _ in range(n_gen):
        # Snapshot elites before the generation is overwritten (GAE only).
        if elitism:
            elites = [toolbox.clone(e) for e in tools.selBest(population, n_elite)]

        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, n_pop)]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_cx:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < p_mut:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring

        # Overwrite the n_elite worst individuals with the preserved elites.
        if elitism:
            worst = sorted(range(n_pop), key=lambda i: population[i].fitness.values[0], reverse=True)
            for slot, elite in zip(worst[:n_elite], elites):
                population[slot] = elite

        fitness_history.append(min(ind.fitness.values[0] for ind in population))

    runtime_s = time.perf_counter() - t0

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
