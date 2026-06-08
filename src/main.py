"""Single-run entry point: run all algorithms on one generated graph."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

from algorithms import aco, ga, sa
from fitness import chromatic_gap
from graph_utils import dsatur, make_random_graph

DEFAULT_N = 50
DEFAULT_P = 0.5
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run all colouring algorithms on one graph.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of vertices")
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="Edge probability")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--k_max", type=int, default=None, help="Max colours (default: n)")
    return parser.parse_args()


def main() -> None:
    """Generate a random graph and run all algorithms, printing a summary table."""
    args = parse_args()
    k_max = args.k_max if args.k_max is not None else args.n

    G = make_random_graph(args.n, args.p, args.seed)
    print(f"Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, p={args.p}, seed={args.seed}")

    dsatur_coloring = dsatur(G)
    chi_ref = len(set(dsatur_coloring.values()))
    print(f"DSATUR reference k={chi_ref}\n")

    runners = [
        ("GA",  ga.run,  {**ga.DEFAULT_PARAMS,  "elitism": False}),
        ("GAE", ga.run,  {**ga.DEFAULT_PARAMS,  "elitism": True}),
        ("ACO", aco.run, dict(aco.DEFAULT_PARAMS)),
        ("SA",  sa.run,  dict(sa.DEFAULT_PARAMS)),
    ]

    print(f"{'Algo':<6} {'k_used':>6} {'violations':>10} {'gap':>5} {'runtime_s':>10}")
    print("-" * 42)
    for name, run_fn, params in runners:
        try:
            result = run_fn(G, k_max, params, seed=args.seed)
            gap = chromatic_gap(result.k_used, chi_ref)
            print(
                f"{name:<6} {result.k_used:>6} {result.violations:>10} "
                f"{gap:>5} {result.runtime_s:>10.3f}"
            )
        except NotImplementedError:
            print(f"{name:<6} {'(not implemented)':>32}")


if __name__ == "__main__":
    main()
