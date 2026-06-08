"""Full benchmark sweep across graph sizes and algorithms."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import aco, ga, sa
from graph_utils import brute_force_timed, dsatur, make_random_graph

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark"
BF_MAX_N = 10

# ── CLAUDE.md default params ──────────────────────────────────────────────────
ALGO_PARAMS: dict[str, dict] = {
    "ga": {
        "n_pop": 100, "p_cx": 0.5, "p_mut": 0.2, "p_ind": 0.05,
        "n_gen": 200, "n_elite": 3, "t_size": 4, "elitism": False,
    },
    "gae": {
        "n_pop": 100, "p_cx": 0.5, "p_mut": 0.2, "p_ind": 0.05,
        "n_gen": 200, "n_elite": 3, "t_size": 4, "elitism": True,
    },
    "aco": {
        "n_ants": 50, "alpha": 1.0, "beta": 3.0, "rho": 0.2,
        "Q": 1.0, "tau_min": 0.01, "n_iter": 300,
    },
    "sa": {
        "T0": 100.0, "gamma": 0.995, "n_step": None, "n_stall": 500, "n_max": None,
    },
}

_RUNNERS: list[tuple[str, object, str]] = [
    ("ga",  ga.run,  "ga"),
    ("gae", ga.run,  "gae"),
    ("aco", aco.run, "aco"),
    ("sa",  sa.run,  "sa"),
]

_FIELDNAMES = [
    "algo", "n", "p", "rep", "seed",
    "k_used", "violations", "gap_dsatur", "gap_bf", "runtime_s",
]


# ── Row builders ──────────────────────────────────────────────────────────────

def _row(
    algo: str, n: int, p: float, rep: int, seed: int,
    k_used: int, violations: int, gap_dsatur: int,
    gap_bf: int | str, runtime_s: float,
) -> dict:
    return {
        "algo": algo, "n": n, "p": p, "rep": rep, "seed": seed,
        "k_used": k_used, "violations": violations,
        "gap_dsatur": gap_dsatur, "gap_bf": gap_bf,
        "runtime_s": f"{runtime_s:.4f}",
    }


def _nan_row(algo: str, n: int, p: float, rep: int, seed: int) -> dict:
    """Row with nan for all numeric result fields (used on exception)."""
    return {
        "algo": algo, "n": n, "p": p, "rep": rep, "seed": seed,
        "k_used": "nan", "violations": "nan",
        "gap_dsatur": "nan", "gap_bf": "nan", "runtime_s": "nan",
    }


# ── Core sweep ────────────────────────────────────────────────────────────────

def run_benchmark(
    ns: list[int],
    ps: list[float],
    reps: int,
    out_path: Path | None = None,
) -> Path:
    """Run the full benchmark sweep and write results to CSV.

    For each (n, p, rep) generates G ~ G(n,p) with seed=rep, runs all
    algorithms, and writes one CSV row per algorithm.  Exceptions are logged
    to stderr and recorded as nan rows so the sweep always completes.

    Args:
        ns: Graph sizes to sweep.
        ps: Edge probabilities to sweep.
        reps: Repetitions per (n, p); seeds are 0..reps-1.
        out_path: Override output CSV path (default: timestamped file).

    Returns:
        Path to the written CSV file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"results_{ts}.csv"

    total = len(ns) * len(ps) * reps
    done = 0

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()

        for n in ns:
            k_max = n
            for p in ps:
                for rep in range(reps):
                    seed = rep
                    t_start = time.perf_counter()
                    G = make_random_graph(n, p, seed)

                    # ── DSATUR (reference; always runs) ───────────────────────
                    t0 = time.perf_counter()
                    ds_col = dsatur(G)
                    ds_time = time.perf_counter() - t0
                    dsatur_k = len(set(ds_col.values()))
                    ds_viol = sum(1 for u, v in G.edges() if ds_col[u] == ds_col[v])

                    # ── Brute force (small n only) ────────────────────────────
                    # Run BF before writing DSATUR row so gap_bf is available.
                    bf_k: int | None = None
                    buffered_bf_row: dict | None = None
                    if n <= BF_MAX_N:
                        try:
                            bf_col, bf_time = brute_force_timed(G, k_max)
                            if bf_col is not None:
                                bf_k = len(set(bf_col.values()))
                                bf_viol = sum(
                                    1 for u, v in G.edges() if bf_col[u] == bf_col[v]
                                )
                                buffered_bf_row = _row(
                                    "bf", n, p, rep, seed,
                                    bf_k, bf_viol, bf_k - dsatur_k, 0, bf_time,
                                )
                        except Exception as exc:
                            logging.error("bf n=%d p=%s rep=%d: %s", n, p, rep, exc)
                            buffered_bf_row = _nan_row("bf", n, p, rep, seed)

                    # Write DSATUR row (bf_k now known)
                    writer.writerow(_row(
                        "dsatur", n, p, rep, seed,
                        dsatur_k, ds_viol, 0,
                        dsatur_k - bf_k if bf_k is not None else "",
                        ds_time,
                    ))
                    if buffered_bf_row is not None:
                        writer.writerow(buffered_bf_row)

                    # ── Metaheuristics ────────────────────────────────────────
                    progress: list[str] = []
                    for algo_name, run_fn, pkey in _RUNNERS:
                        try:
                            result = run_fn(  # type: ignore[operator]
                                G, k_max, ALGO_PARAMS[pkey], seed=seed
                            )
                            gap_ds = result.k_used - dsatur_k
                            gap_bf = result.k_used - bf_k if bf_k is not None else ""
                            writer.writerow(_row(
                                algo_name, n, p, rep, seed,
                                result.k_used, result.violations,
                                gap_ds, gap_bf, result.runtime_s,
                            ))
                            progress.append(f"{algo_name}={result.k_used}({gap_ds:+d})")
                        except Exception as exc:
                            logging.error(
                                "%s n=%d p=%s rep=%d: %s", algo_name, n, p, rep, exc
                            )
                            writer.writerow(_nan_row(algo_name, n, p, rep, seed))
                            progress.append(f"{algo_name}=err")

                    fh.flush()
                    done += 1
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"n={n:>3}  p={p:.1f}  rep={rep:>2}/{reps}"
                        f"  dsatur={dsatur_k}  {'  '.join(progress)}"
                        f"  [{elapsed:.1f}s]  ({done}/{total})",
                        flush=True,
                    )

    print(f"\nDone. Results → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Full benchmark sweep.")
    p.add_argument("--ns",   type=int,   nargs="+",
                   default=[20, 30, 40, 50, 75, 100, 150, 200])
    p.add_argument("--reps", type=int,   default=20)
    p.add_argument("--ps",   type=float, nargs="+", default=[0.5])
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    args = parse_args()
    run_benchmark(args.ns, args.ps, args.reps)
