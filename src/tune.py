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
from config import ALGO_PARAMS as BASE_PARAMS
from graph_utils import dsatur, make_random_graph

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tuning"
TUNE_P = 0.5

# OFAT sweep definitions: list of (varied_param, values).
# "alpha_beta" is a joint sweep; values are (alpha, beta) tuples.
#
# Range rationale:
#   GA/GAE p_mut: first run (n=20, few reps) was monotone at 0.3 → shifted up.
#   GA/GAE p_cx:  first run was monotone at 0.8 → shifted up; 1.0 is valid
#                 (DEAP applies cxTwoPoint only when rand < p_cx, so 1.0 = always).
#   GA/GAE p_ind: no prior data; [0.01,0.2] spans the meaningful range for
#                 per-gene mutation on chromosomes of length n≈40–60.
#   GA/GAE t_size: sweet-spot at 4 confirmed; dropped t_size=2 (consistently worst).
#   ACO n_ants:   first run was monotone at 100 → extended upper end.
#   ACO rho:      no monotone trend; [0.1,0.5] kept.
#   ACO alpha_beta: (1,5) was best; extended beta range to confirm plateau.
#   SA gamma:     wide range kept; no boundary issue observed.
#   SA T0:        low T0 (10) appeared best on n=20 but data was too thin
#                 to trust → re-swept on proper n range.
#   SA n_stall:   no prior data; [100,1000] spans the meaningful range.
SWEEPS: dict[str, list[tuple[str, list]]] = {
    "ga": [
        ("p_mut",  [0.2, 0.3, 0.4, 0.5]),        # shifted up: was monotone at 0.3
        ("p_cx",   [0.7, 0.8, 0.9, 1.0]),         # shifted up: was monotone at 0.8
        ("p_ind",  [0.01, 0.05, 0.1, 0.2]),
        ("t_size", [3, 4, 5, 6]),                  # dropped t_size=2 (worst in prior run)
    ],
    "gae": [
        ("p_mut",   [0.2, 0.3, 0.4, 0.5]),        # same reasoning as GA
        ("p_cx",    [0.7, 0.8, 0.9, 1.0]),
        ("p_ind",   [0.01, 0.05, 0.1, 0.2]),
        ("n_elite", [1, 3, 5, 10]),
        ("t_size",  [3, 4, 5, 6]),
    ],
    "aco": [
        ("rho",        [0.1, 0.2, 0.3, 0.5]),
        ("alpha_beta", [(1.0, 3.0), (1.0, 5.0), (1.0, 7.0), (2.0, 5.0)]),  # extend beta
        ("n_ants",     [50, 100, 150, 200]),       # shifted up: was monotone at 100
    ],
    "sa": [
        ("gamma",  [0.90, 0.95, 0.99, 0.999]),
        ("T0",     [10.0, 50.0, 100.0, 200.0]),   # re-sweep with proper n range
        ("n_stall", [100, 250, 500, 1000]),
    ],
}

_RUN: dict[str, object] = {"ga": ga.run, "gae": ga.run, "aco": aco.run, "sa": sa.run}


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


def flat_combos(algo: str) -> list[tuple[str, object]]:
    """Return all (varied_param, value) pairs for algo in SWEEPS order.

    Used to map a flat integer index to a specific (param, value) combo so
    that SLURM array tasks can each run exactly one combination.

    Args:
        algo: Algorithm key.

    Returns:
        Ordered list of (varied_param, value) tuples.
    """
    return [(p, v) for p, vals in SWEEPS[algo] for v in vals]


# ── Core sweep ────────────────────────────────────────────────────────────────

def run_tune(
    algo: str,
    ns: list[int],
    reps: int,
    out_path: Path | None = None,
    combo_idx: int | None = None,
) -> tuple[Path, dict[tuple[str, str], list[float]]]:
    """Run OFAT sweep (or a single combo) and write results to CSV.

    For each (varied_param, value) combination, each n in ns, and each rep in
    range(reps), generates G(n, p=0.5, seed=rep), runs the algorithm and
    DSATUR, records the gap.

    Args:
        algo: Algorithm to sweep (ga, gae, aco, sa).
        ns: Graph sizes to include.
        reps: Repetitions per (n, varied_param, param_value).
        out_path: Override output CSV path (default: auto-timestamped).
        combo_idx: If given, run only this flat index from flat_combos(algo).
            Output filename includes the index so cluster chunks don't clash.

    Returns:
        (csv_path, gaps_by_setting) where gaps_by_setting maps
        (varied_param, pval_str) -> list of gap floats for the summary table.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    combos: list[tuple[str, object]]
    if combo_idx is not None:
        all_combos = flat_combos(algo)
        if not 0 <= combo_idx < len(all_combos):
            raise ValueError(
                f"combo_idx {combo_idx} out of range for {algo} "
                f"(0–{len(all_combos) - 1})"
            )
        combos = [all_combos[combo_idx]]
        if out_path is None:
            out_path = RESULTS_DIR / f"{algo}_tuning_chunk{combo_idx:03d}_{ts}.csv"
    else:
        combos = flat_combos(algo)
        if out_path is None:
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

        for varied_param, value in combos:
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
    p.add_argument("--algo",  required=True, choices=["ga", "gae", "aco", "sa"])
    p.add_argument("--reps",  type=int, default=10, help="Repetitions per setting")
    p.add_argument("--ns",    type=int, nargs="+", default=[40, 50, 60],
                   help="Graph sizes (space-separated)")
    p.add_argument("--combo", type=int, default=None,
                   help="Run only this flat (param, value) index (for cluster arrays). "
                        "Use --list-combos to see the index mapping.")
    p.add_argument("--list-combos", action="store_true",
                   help="Print (index, param, value) table for --algo and exit.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_combos:
        print(f"Flat combo indices for --algo {args.algo}:")
        for i, (p, v) in enumerate(flat_combos(args.algo)):
            print(f"  {i:3d}  {p}={v}")
        raise SystemExit(0)

    if args.combo is not None:
        varied_param, value = flat_combos(args.algo)[args.combo]
        print(
            f"Single combo: algo={args.algo}  [{args.combo}] {varied_param}={value}"
            f"  ns={args.ns}  reps={args.reps}\n"
        )
        csv_path, gaps = run_tune(args.algo, args.ns, args.reps, combo_idx=args.combo)
        _print_summary(args.algo, gaps)
    else:
        n_combos = len(flat_combos(args.algo))
        total = n_combos * len(args.ns) * args.reps
        print(
            f"Full sweep: algo={args.algo}  ns={args.ns}  reps={args.reps}"
            f"  combos={n_combos}  total_runs={total}\n"
        )
        csv_path, gaps = run_tune(args.algo, args.ns, args.reps)
        _print_summary(args.algo, gaps)
