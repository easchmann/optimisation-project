"""Benchmark sweep over fixed DIMACS instances with known chi(G)."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import aco, ga, hybrid_aco_sa, hybrid_ga_sa, sa
from config import ALGO_PARAMS
from graph_utils import dsatur, load_chromatic_numbers, load_dimacs

DATA_DIR = Path(__file__).parent.parent / "data" / "dimacs"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "dimacs"

_RUNNERS: list[tuple[str, object, str]] = [
    ("ga",  ga.run,  "ga"),
    ("gae", ga.run,  "gae"),
    ("aco", aco.run, "aco"),
    ("sa",  sa.run,  "sa"),
    ("hybrid_ga_sa",  hybrid_ga_sa.run,  "hybrid_ga_sa"),
    ("hybrid_aco_sa", hybrid_aco_sa.run, "hybrid_aco_sa"),
]

_FIELDNAMES = [
    "algo", "instance", "n", "m", "chi_true", "seed",
    "k_used", "violations", "gap_true", "gap_dsatur", "runtime_s",
]


# ── Row builders ──────────────────────────────────────────────────────────────

def _row(
    algo: str, instance: str, n: int, m: int, chi_true: int, seed: int | str,
    k_used: int, violations: int, gap_true: int, gap_dsatur: int, runtime_s: float,
) -> dict:
    return {
        "algo": algo, "instance": instance, "n": n, "m": m, "chi_true": chi_true,
        "seed": seed, "k_used": k_used, "violations": violations,
        "gap_true": gap_true, "gap_dsatur": gap_dsatur,
        "runtime_s": f"{runtime_s:.4f}",
    }


def _nan_row(algo: str, instance: str, n: int, m: int, chi_true: int, seed: int) -> dict:
    """Row with nan for all numeric result fields (used on exception)."""
    return {
        "algo": algo, "instance": instance, "n": n, "m": m, "chi_true": chi_true,
        "seed": seed, "k_used": "nan", "violations": "nan",
        "gap_true": "nan", "gap_dsatur": "nan", "runtime_s": "nan",
    }


# ── Core sweep ────────────────────────────────────────────────────────────────

def run_dimacs_benchmark(
    instances: list[str] | None,
    reps: int,
    out_path: Path | None = None,
) -> Path:
    """Run all algorithms against fixed DIMACS instances and write results to CSV.

    Unlike the random-graph sweep, the graph is fixed per instance, so DSATUR
    (deterministic) is computed once per instance rather than once per rep;
    each metaheuristic still runs `reps` times with seed = rep index.

    Args:
        instances: Filename stems to include (default: every .col file in
            data/dimacs/).
        reps: Repetitions per (instance, algorithm); seeds are 0..reps-1.
        out_path: Override output CSV path (default: timestamped file).

    Returns:
        Path to the written CSV file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"dimacs_results_{ts}.csv"

    chi_lookup = load_chromatic_numbers(DATA_DIR / "chromatic_numbers.json")

    col_files = sorted(DATA_DIR.glob("*.col"))
    if instances is not None:
        wanted = set(instances)
        col_files = [f for f in col_files if f.stem in wanted]

    total = len(col_files) * len(_RUNNERS) * reps
    done = 0

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()

        for col_path in col_files:
            instance = col_path.stem
            chi_true = chi_lookup.get(instance)
            if chi_true is None:
                logging.error("%s: no chromatic_numbers.json entry, skipped", instance)
                continue

            G = load_dimacs(col_path)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            k_max = n

            # ── DSATUR (reference; deterministic, computed once per instance) ──
            t0 = time.perf_counter()
            ds_col = dsatur(G)
            ds_time = time.perf_counter() - t0
            dsatur_k = len(set(ds_col.values()))
            ds_viol = sum(1 for u, v in G.edges() if ds_col[u] == ds_col[v])
            if ds_viol:
                logging.error("DSATUR produced %d violations on %s", ds_viol, instance)

            writer.writerow(_row(
                "dsatur", instance, n, m, chi_true, "",
                dsatur_k, ds_viol, dsatur_k - chi_true, 0, ds_time,
            ))

            # ── Metaheuristics ────────────────────────────────────────────────
            for algo_name, run_fn, pkey in _RUNNERS:
                for seed in range(reps):
                    t_start = time.perf_counter()
                    try:
                        result = run_fn(  # type: ignore[operator]
                            G, k_max, ALGO_PARAMS[pkey], seed=seed
                        )
                        gap_true = result.k_used - chi_true
                        gap_ds = result.k_used - dsatur_k
                        writer.writerow(_row(
                            algo_name, instance, n, m, chi_true, seed,
                            result.k_used, result.violations,
                            gap_true, gap_ds, result.runtime_s,
                        ))
                        status = f"{result.k_used}({gap_true:+d})"
                    except Exception as exc:
                        logging.error(
                            "%s %s seed=%d: %s", instance, algo_name, seed, exc
                        )
                        writer.writerow(_nan_row(algo_name, instance, n, m, chi_true, seed))
                        status = "err"

                    fh.flush()
                    done += 1
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"{instance:<10} chi={chi_true:>2}  dsatur={dsatur_k:>2}"
                        f"  {algo_name:<13} seed={seed:>2}/{reps}"
                        f"  {status}  [{elapsed:.1f}s]  ({done}/{total})",
                        flush=True,
                    )

    print(f"\nDone. Results → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="DIMACS instance benchmark sweep.")
    p.add_argument(
        "--instances", type=str, nargs="+", default=None,
        help="Instance stems to run (default: every .col file in data/dimacs/)",
    )
    p.add_argument("--reps", type=int, default=10, help="Repetitions per (instance, algorithm)")
    p.add_argument("--out",  type=Path, default=None,
                   help="Output CSV path (default: auto-timestamped)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    args = parse_args()
    run_dimacs_benchmark(args.instances, args.reps, out_path=args.out)
