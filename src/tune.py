"""OFAT hyperparameter sweep for a single algorithm."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import aco, ga, sa
from graph_utils import dsatur, make_random_graph

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tuning"
TUNE_P = 0.5

# ── Defaults from CLAUDE.md ───────────────────────────────────────────────────
BASE_PARAMS: dict[str, dict] = {
    "ga": {
        "n_pop": 100, "p_cx": 0.5, "p_mut": 0.2, "p_ind": 0.05,
        "n_gen": 200, "n_elite": 3, "t_size": 4, "elitism": False,
    },
    "aco": {
        "n_ants": 50, "alpha": 1.0, "beta": 3.0, "rho": 0.2,
        "Q": 1.0, "tau_min": 0.01, "n_iter": 300,
    },
    "sa": {
        "T0": 100.0, "gamma": 0.995, "n_step": None, "n_stall": 500, "n_max": None,
    },
}

# OFAT sweep definitions: list of (varied_param, values).
# "alpha_beta" is a joint sweep; values are (alpha, beta) tuples.
SWEEPS: dict[str, list[tuple[str, list]]] = {
    "ga": [
        ("p_mut",  [0.05, 0.1, 0.2, 0.3]),
        ("p_cx",   [0.4, 0.6, 0.8]),
        ("t_size", [2, 3, 4, 6]),
    ],
    "aco": [
        ("rho",        [0.1, 0.2, 0.3, 0.5]),
        ("alpha_beta", [(1.0, 2.0), (1.0, 3.0), (2.0, 3.0), (1.0, 5.0)]),
        ("n_ants",     [20, 50, 100]),
    ],
    "sa": [
        ("gamma", [0.90, 0.95, 0.99, 0.999]),
        ("T0",    [10.0, 50.0, 200.0, 500.0]),
    ],
}

_RUN: dict[str, object] = {"ga": ga.run, "aco": aco.run, "sa": sa.run}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_params(algo: str, varied_param: str, value: object) -> dict:
    """Return base params with one setting overridden.

    Args:
        algo: Algorithm key.
        varied_param: Parameter name, or "alpha_beta" for a joint override.
        value: New value; a (alpha, beta) tuple when varied_param=="alpha_beta".

    Returns:
        Full params dict ready to pass to run().
    """
    params = dict(BASE_PARAMS[algo])
    if varied_param == "alpha_beta":
        params["alpha"], params["beta"] = value  # type: ignore[misc]
    else:
        params[varied_param] = value
    return params


def _pval_str(varied_param: str, value: object) -> str:
    """Canonical string for a param value used in CSV and the summary table."""
    if varied_param == "alpha_beta":
        a, b = value  # type: ignore[misc]
        return f"({a},{b})"
    return str(value)


# ── Core sweep ────────────────────────────────────────────────────────────────

def run_tune(
    algo: str,
    ns: list[int],
    reps: int,
    out_path: Path | None = None,
) -> tuple[Path, dict[tuple[str, str], list[float]]]:
    """Run OFAT sweep and write results to CSV.

    For each (varied_param, value) in SWEEPS[algo], each n in ns, and each rep
    in range(reps), generates a fresh G(n, p=0.5) with seed=rep, runs the
    algorithm and DSATUR, records the gap.

    Args:
        algo: Algorithm to sweep (ga, aco, sa).
        ns: Graph sizes to include.
        reps: Repetitions per (n, varied_param, param_value).
        out_path: Override output CSV path (default: timestamped file).

    Returns:
        (csv_path, gaps_by_setting) where gaps_by_setting maps
        (varied_param, pval_str) -> list of gap floats for the summary table.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{algo}_tuning_{ts}.csv"

    run_fn = _RUN[algo]
    fieldnames = [
        "algo", "n", "varied_param", "param_value", "rep",
        "k_used", "dsatur_k", "gap", "runtime_s",
    ]
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for varied_param, values in SWEEPS[algo]:
            for value in values:
                pval = _pval_str(varied_param, value)
                params = _make_params(algo, varied_param, value)
                print(f"  {varied_param}={pval} ...", end="", flush=True)
                count = 0

                for n in ns:
                    for rep in range(reps):
                        G = make_random_graph(n, TUNE_P, seed=rep)
                        dsatur_k = len(set(dsatur(G).values()))
                        result = run_fn(G, n, params, seed=rep)  # type: ignore[operator]
                        gap = result.k_used - dsatur_k
                        gaps[(varied_param, pval)].append(float(gap))
                        writer.writerow({
                            "algo": algo, "n": n,
                            "varied_param": varied_param, "param_value": pval,
                            "rep": rep, "k_used": result.k_used,
                            "dsatur_k": dsatur_k, "gap": gap,
                            "runtime_s": f"{result.runtime_s:.4f}",
                        })
                        count += 1
                fh.flush()
                print(f" done ({count} runs)")

    print(f"\nResults written to {out_path}")
    return out_path, gaps


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(algo: str, gaps: dict[tuple[str, str], list[float]]) -> None:
    """Print mean gap per (varied_param, param_value) grouped by varied_param.

    Args:
        algo: Algorithm name (used in the header).
        gaps: Mapping from (varied_param, pval_str) to list of gap values.
    """
    W = (16, 16, 10)
    bar = "-" * (sum(W) + 6)
    print(f"\n── {algo.upper()} tuning summary {'─' * (len(bar) - len(algo) - 19)}")
    print(f"  {'varied_param':<{W[0]}}  {'param_value':<{W[1]}}  {'mean_gap':>{W[2]}}")
    print(f"  {bar}")
    prev = None
    for (varied_param, pval), gap_list in gaps.items():
        if prev is not None and prev != varied_param:
            print(f"  {'':─<{sum(W) + 6}}")
        mean = sum(gap_list) / len(gap_list)
        print(f"  {varied_param:<{W[0]}}  {pval:<{W[1]}}  {mean:>{W[2]}.3f}")
        prev = varied_param
    print(f"  {bar}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="OFAT hyperparameter sweep.")
    p.add_argument("--algo", required=True, choices=["ga", "aco", "sa"])
    p.add_argument("--reps", type=int, default=10, help="Repetitions per setting")
    p.add_argument("--ns",   type=int, nargs="+", default=[40, 50, 60],
                   help="Graph sizes (space-separated)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_combos = sum(len(vals) for _, vals in SWEEPS[args.algo])
    total = n_combos * len(args.ns) * args.reps
    print(
        f"Sweep: algo={args.algo}  ns={args.ns}  reps={args.reps}"
        f"  combos={n_combos}  total_runs={total}\n"
    )
    csv_path, gaps = run_tune(args.algo, args.ns, args.reps)
    _print_summary(args.algo, gaps)
