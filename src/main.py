"""Single-run entry point: colour one random graph with selected algorithms."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

from algorithms import aco, ga, sa
from fitness import AlgoResult, chromatic_gap
from graph_utils import brute_force_timed, dsatur, make_random_graph

# ── Tuned defaults from CLAUDE.md defaults table ─────────────────────────────
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
        "n_ants": 50, "alpha": 1.0, "beta": 3.0, "rho": 0.2, "n_iter": 300,
    },
    "sa": {
        "T0": 100.0, "gamma": 0.995, "n_step": None, "n_stall": 500,
    },
}

BF_MAX_N = 15

# Column widths: Algorithm | Colours | Violations | Gap vs DSATUR | Runtime (s)
_W = (12, 9, 11, 15, 12)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _divider() -> str:
    return "-" * (sum(_W) + (len(_W) - 1) * 2)


def _header() -> str:
    labels = ["Algorithm", "Colours", "Violations", "Gap vs DSATUR", "Runtime (s)"]
    return "  ".join(
        lbl.ljust(_W[0]) if i == 0 else lbl.rjust(_W[i])
        for i, lbl in enumerate(labels)
    )


def _row(label: str, result: AlgoResult, gap: int) -> str:
    """Format one result row; appends [INFEASIBLE] when violations > 0."""
    marker = "  [INFEASIBLE]" if result.violations > 0 else ""
    return (
        f"{label:<{_W[0]}}"
        f"  {result.k_used:>{_W[1]}}"
        f"  {result.violations:>{_W[2]}}"
        f"  {gap:>+{_W[3]}}"
        f"  {result.runtime_s:>{_W[4]}.3f}"
        f"{marker}"
    )


# ── Algorithm wrappers ────────────────────────────────────────────────────────

def _run_dsatur(G: nx.Graph) -> AlgoResult:
    """Wrap DSATUR in an AlgoResult."""
    t0 = time.perf_counter()
    coloring = dsatur(G)
    runtime_s = time.perf_counter() - t0
    k_used = len(set(coloring.values()))
    violations = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])
    return AlgoResult(
        coloring=coloring, k_used=k_used, violations=violations, runtime_s=runtime_s
    )


def _run_bf(G: nx.Graph, k_max: int) -> AlgoResult | None:
    """Run brute force; returns None when no colouring found within k_max."""
    coloring, runtime_s = brute_force_timed(G, k_max)
    if coloring is None:
        return None
    k_used = len(set(coloring.values()))
    violations = sum(1 for u, v in G.edges() if coloring[u] == coloring[v])
    return AlgoResult(
        coloring=coloring, k_used=k_used, violations=violations, runtime_s=runtime_s
    )


# ── Core runner ───────────────────────────────────────────────────────────────

def _collect_rows(
    algo: str,
    G: nx.Graph,
    k_max: int,
    seed: int,
    dsatur_ref: AlgoResult,
) -> list[tuple[str, AlgoResult | None, int | None, str | None]]:
    """Build ordered list of (label, result, gap, skip_message).

    skip_message is non-None only when a result is intentionally omitted.
    """
    n = G.number_of_nodes()
    dsatur_k = dsatur_ref.k_used
    targets = ["dsatur", "bf", "ga", "gae", "aco", "sa"] if algo == "all" else [algo]

    rows: list[tuple[str, AlgoResult | None, int | None, str | None]] = []

    for name in targets:
        if name == "dsatur":
            rows.append(("DSATUR", dsatur_ref, 0, None))

        elif name == "bf":
            if n > BF_MAX_N:
                rows.append(("BF", None, None, f"n={n} > BF_MAX_N={BF_MAX_N}"))
            else:
                result = _run_bf(G, k_max)
                if result is None:
                    rows.append(("BF", None, None, f"no {k_max}-colouring found"))
                else:
                    rows.append(("BF", result, chromatic_gap(result.k_used, dsatur_k), None))

        elif name == "ga":
            result = ga.run(G, k_max, ALGO_PARAMS["ga"], seed=seed)
            rows.append(("GA", result, chromatic_gap(result.k_used, dsatur_k), None))

        elif name == "gae":
            result = ga.run(G, k_max, ALGO_PARAMS["gae"], seed=seed)
            rows.append(("GAE", result, chromatic_gap(result.k_used, dsatur_k), None))

        elif name == "aco":
            result = aco.run(G, k_max, ALGO_PARAMS["aco"], seed=seed)
            rows.append(("ACO", result, chromatic_gap(result.k_used, dsatur_k), None))

        elif name == "sa":
            result = sa.run(G, k_max, ALGO_PARAMS["sa"], seed=seed)
            rows.append(("SA", result, chromatic_gap(result.k_used, dsatur_k), None))

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Graph colouring via metaheuristics.")
    p.add_argument("--n",    type=int,   default=50,    help="Number of vertices")
    p.add_argument("--p",    type=float, default=0.5,   help="Edge probability")
    p.add_argument("--seed", type=int,   default=42,    help="Random seed")
    p.add_argument("--k_max", type=int,  default=20,    help="Maximum colours")
    p.add_argument(
        "--algo", default="all",
        choices=["ga", "gae", "aco", "sa", "dsatur", "bf", "all"],
        help="Algorithm to run (default: all)",
    )
    return p.parse_args()


def main() -> None:
    """Generate a random graph and print an algorithm comparison table."""
    args = parse_args()

    print(f"Seed: {args.seed}")
    G = make_random_graph(args.n, args.p, args.seed)
    print(f"Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, p={args.p}")

    dsatur_ref = _run_dsatur(G)
    print(f"DSATUR reference: k={dsatur_ref.k_used}\n")

    rows = _collect_rows(args.algo, G, args.k_max, args.seed, dsatur_ref)

    print(_header())
    print(_divider())
    for label, result, gap, skip_msg in rows:
        if skip_msg is not None:
            print(f"  ({label})  WARNING: skipped — {skip_msg}")
        else:
            assert result is not None and gap is not None
            print(_row(label, result, gap))


if __name__ == "__main__":
    main()
