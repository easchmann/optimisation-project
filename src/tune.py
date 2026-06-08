"""Hyperparameter sweep for a single algorithm."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from itertools import product as cartesian
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import aco, ga, sa
from fitness import chromatic_gap
from graph_utils import dsatur, make_random_graph

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tuning"

# Parameter grids per algorithm
GRIDS: dict[str, dict] = {
    "ga": {
        "n_pop":  [50, 100, 200],
        "p_cx":   [0.3, 0.5, 0.7],
        "p_mut":  [0.1, 0.2, 0.3],
        "n_gen":  [100, 200],
        "elitism": [False],
    },
    "gae": {
        "n_pop":   [50, 100, 200],
        "p_cx":    [0.3, 0.5, 0.7],
        "p_mut":   [0.1, 0.2, 0.3],
        "n_gen":   [100, 200],
        "n_elite": [1, 3, 5],
        "elitism": [True],
    },
    "aco": {
        "n_ants": [20, 50, 100],
        "alpha":  [0.5, 1.0, 2.0],
        "beta":   [1.0, 3.0, 5.0],
        "rho":    [0.1, 0.2, 0.4],
        "n_iter": [150, 300],
    },
    "sa": {
        "T0":    [50.0, 100.0, 200.0],
        "gamma": [0.990, 0.995, 0.999],
        "n_stall": [300, 500, 1000],
    },
}

TUNE_N = 50
TUNE_P = 0.5
TUNE_REPS = 5


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for one algorithm.")
    parser.add_argument("--algo", required=True, choices=list(GRIDS.keys()))
    parser.add_argument("--n", type=int, default=TUNE_N)
    parser.add_argument("--p", type=float, default=TUNE_P)
    parser.add_argument("--reps", type=int, default=TUNE_REPS)
    return parser.parse_args()


def run_tune(algo: str, n: int, p: float, reps: int, out_path: Path | None = None) -> Path:
    """Run grid search and write CSV.

    Args:
        algo: Algorithm name key.
        n: Graph size.
        p: Edge probability.
        reps: Number of random graph repetitions per parameter combination.
        out_path: Override output path.

    Returns:
        Path to the written CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{algo}_tuning_{ts}.csv"

    grid = GRIDS[algo]
    keys = list(grid.keys())
    combos = list(cartesian(*[grid[k] for k in keys]))

    run_fn = {"ga": ga.run, "gae": ga.run, "aco": aco.run, "sa": sa.run}[algo]
    base_params = {
        "ga":  {**ga.DEFAULT_PARAMS,  "elitism": False},
        "gae": {**ga.DEFAULT_PARAMS,  "elitism": True},
        "aco": dict(aco.DEFAULT_PARAMS),
        "sa":  dict(sa.DEFAULT_PARAMS),
    }[algo]

    fieldnames = keys + ["rep", "n", "p", "k_used", "violations", "gap_dsatur", "runtime_s"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for combo in combos:
            params = {**base_params, **dict(zip(keys, combo))}
            for rep in range(reps):
                G = make_random_graph(n, p, seed=rep)
                dsatur_k = len(set(dsatur(G).values()))
                try:
                    result = run_fn(G, n, params, seed=rep)
                except NotImplementedError:
                    continue
                row = dict(zip(keys, combo))
                row.update({
                    "rep": rep, "n": n, "p": p,
                    "k_used": result.k_used,
                    "violations": result.violations,
                    "gap_dsatur": chromatic_gap(result.k_used, dsatur_k),
                    "runtime_s": f"{result.runtime_s:.4f}",
                })
                writer.writerow(row)

    print(f"Tuning results written to {out_path}")
    return out_path


if __name__ == "__main__":
    args = parse_args()
    run_tune(args.algo, args.n, args.p, args.reps)
