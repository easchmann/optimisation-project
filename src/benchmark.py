"""Full benchmark sweep across graph sizes and algorithms."""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

from algorithms import aco, ga, sa
from fitness import chromatic_gap
from graph_utils import brute_force, dsatur, make_random_graph

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark"

N_VALUES = list(range(20, 210, 10))
P_VALUES = [0.3, 0.5, 0.7]
N_REPS = 20
BF_MAX_N = 20  # brute force only run for small graphs


def run_benchmark(out_path: Path | None = None) -> Path:
    """Run the full benchmark sweep and write results to CSV.

    Args:
        out_path: Override output path (defaults to timestamped file in results/benchmark/).

    Returns:
        Path to the written CSV file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"results_{ts}.csv"

    runners = [
        ("ga",  ga.run,  {**ga.DEFAULT_PARAMS,  "elitism": False}),
        ("gae", ga.run,  {**ga.DEFAULT_PARAMS,  "elitism": True}),
        ("aco", aco.run, dict(aco.DEFAULT_PARAMS)),
        ("sa",  sa.run,  dict(sa.DEFAULT_PARAMS)),
    ]

    fieldnames = ["algo", "n", "p", "seed", "k_used", "violations", "gap_dsatur", "gap_bf", "runtime_s"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for n in N_VALUES:
            for p in P_VALUES:
                for rep in range(N_REPS):
                    seed = rep
                    G = make_random_graph(n, p, seed)
                    k_max = n

                    dsatur_k = len(set(dsatur(G).values()))

                    bf_k: int | None = None
                    if n <= BF_MAX_N:
                        bf_coloring = brute_force(G, k_max)
                        if bf_coloring is not None:
                            bf_k = len(set(bf_coloring.values()))

                    for algo_name, run_fn, params in runners:
                        try:
                            result = run_fn(G, k_max, params, seed=seed)
                        except NotImplementedError:
                            continue

                        gap_dsatur = chromatic_gap(result.k_used, dsatur_k)
                        gap_bf = chromatic_gap(result.k_used, bf_k) if bf_k is not None else None

                        writer.writerow({
                            "algo": algo_name,
                            "n": n,
                            "p": p,
                            "seed": seed,
                            "k_used": result.k_used,
                            "violations": result.violations,
                            "gap_dsatur": gap_dsatur,
                            "gap_bf": gap_bf if gap_bf is not None else "",
                            "runtime_s": f"{result.runtime_s:.4f}",
                        })

    print(f"Results written to {out_path}")
    return out_path


if __name__ == "__main__":
    run_benchmark()
