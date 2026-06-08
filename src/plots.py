"""Generate all result figures from the benchmark CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import pandas as pd

from algorithms import aco, ga, sa
from graph_utils import dsatur, make_random_graph

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
BENCHMARK_DIR = Path(__file__).parent.parent / "results" / "benchmark"
DPI = 300

# ── Consistent style per algorithm across all figures ─────────────────────────
ALGO_COLORS: dict[str, str] = {
    "ga": "#1f77b4", "gae": "#ff7f0e", "aco": "#2ca02c",
    "sa": "#d62728", "dsatur": "#9467bd", "bf": "#8c564b",
}
ALGO_MARKERS: dict[str, str] = {
    "ga": "o", "gae": "s", "aco": "^", "sa": "D", "dsatur": "x", "bf": "+",
}
META_ALGOS = ["ga", "gae", "aco", "sa"]   # metaheuristics only (exclude baselines)

_ALGO_PARAMS: dict[str, dict] = {
    "ga":  {"n_pop": 100, "p_cx": 0.5, "p_mut": 0.2, "p_ind": 0.05,
            "n_gen": 200, "n_elite": 3, "t_size": 4, "elitism": False},
    "gae": {"n_pop": 100, "p_cx": 0.5, "p_mut": 0.2, "p_ind": 0.05,
            "n_gen": 200, "n_elite": 3, "t_size": 4, "elitism": True},
    "aco": {"n_ants": 50, "alpha": 1.0, "beta": 3.0, "rho": 0.2,
            "Q": 1.0, "tau_min": 0.01, "n_iter": 300},
    "sa":  {"T0": 100.0, "gamma": 0.995, "n_step": None, "n_stall": 500, "n_max": None},
}
_RUNNERS = [("ga", ga.run, "ga"), ("gae", ga.run, "gae"),
            ("aco", aco.run, "aco"), ("sa", sa.run, "sa")]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _latest_csv(directory: Path) -> Path:
    """Return the CSV with the lexicographically largest name (latest timestamp)."""
    csvs = sorted(directory.glob("*.csv"), key=lambda p: p.name)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csvs[-1]


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent title, axis labels, and grid to an Axes object."""
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True)


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    """Save figure at DPI, bbox tight (preserves outside-right legends)."""
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 1: avg_gap_vs_n ────────────────────────────────────────────────────

def plot_avg_gap_vs_n(df: pd.DataFrame, out_dir: Path) -> None:
    """avg_gap_vs_n.png: mean gap_dsatur ±1 std per algo vs n."""
    sub = df[df["algo"].isin(META_ALGOS)].copy()
    sub["gap_dsatur"] = pd.to_numeric(sub["gap_dsatur"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in META_ALGOS:
        g = sub[sub["algo"] == algo].groupby("n")["gap_dsatur"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(),
                    label=algo.upper(), color=ALGO_COLORS[algo],
                    marker=ALGO_MARKERS[algo], capsize=3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Average Chromatic Gap vs. Problem Size", "n", "Mean gap vs. DSATUR")
    _save(fig, "avg_gap_vs_n.png", out_dir)


# ── Figure 2: gap_vs_bf_small_n ───────────────────────────────────────────────

def plot_gap_vs_bf_small_n(df: pd.DataFrame, out_dir: Path) -> None:
    """gap_vs_bf_small_n.png: mean gap_bf for rows where gap_bf is numeric."""
    sub = df[df["algo"].isin(META_ALGOS)].copy()
    sub["gap_bf"] = pd.to_numeric(sub["gap_bf"], errors="coerce")
    sub = sub.dropna(subset=["gap_bf"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in META_ALGOS:
        g = sub[sub["algo"] == algo].groupby("n")["gap_bf"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(),
                    label=algo.upper(), color=ALGO_COLORS[algo],
                    marker=ALGO_MARKERS[algo], capsize=3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Gap vs. Optimal (BF) for Small n", "n", "Mean gap vs. BF (optimal)")
    _save(fig, "gap_vs_bf_small_n.png", out_dir)


# ── Figure 3: avg_runtime ─────────────────────────────────────────────────────

def plot_avg_runtime(df: pd.DataFrame, out_dir: Path) -> None:
    """avg_runtime.png: mean runtime (log y) for all algorithms including baselines."""
    sub = df.copy()
    sub["runtime_s"] = pd.to_numeric(sub["runtime_s"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ["dsatur", "bf"] + META_ALGOS:
        g = sub[sub["algo"] == algo].groupby("n")["runtime_s"].mean()
        if g.empty:
            continue
        ax.plot(g.index, g, label=algo.upper(),
                color=ALGO_COLORS.get(algo, "gray"),
                marker=ALGO_MARKERS.get(algo, "o"))
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Average Runtime Comparison", "n", "Runtime (s, log scale)")
    _save(fig, "avg_runtime.png", out_dir)


# ── Figure 4: convergence_n100 ────────────────────────────────────────────────

def run_convergence_experiment() -> dict[str, list[float]]:
    """Run each algorithm on G(n=100, p=0.5, seed=0) and return fitness histories.

    Returns:
        Mapping from algo name to fitness_history list.
    """
    G = make_random_graph(100, 0.5, seed=0)
    histories: dict[str, list[float]] = {}
    for name, run_fn, pkey in _RUNNERS:
        result = run_fn(G, 100, _ALGO_PARAMS[pkey], seed=0)  # type: ignore[operator]
        histories[name] = result.fitness_history
    return histories


def plot_convergence_n100(out_dir: Path) -> None:
    """convergence_n100.png: fitness history per algo on G(n=100, p=0.5, seed=0)."""
    print("Running convergence experiment on G(n=100, p=0.5, seed=0) ...")
    histories = run_convergence_experiment()
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, hist in histories.items():
        ax.plot(hist, label=algo.upper(), color=ALGO_COLORS[algo], alpha=0.85)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Convergence at n=100", "Generation / Iteration", "Best Fitness F")
    _save(fig, "convergence_n100.png", out_dir)


# ── Figure 5: std_dev_gap ─────────────────────────────────────────────────────

def plot_std_dev_gap(df: pd.DataFrame, out_dir: Path) -> None:
    """std_dev_gap.png: std dev of gap_dsatur per algo vs n."""
    sub = df[df["algo"].isin(META_ALGOS)].copy()
    sub["gap_dsatur"] = pd.to_numeric(sub["gap_dsatur"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in META_ALGOS:
        g = sub[sub["algo"] == algo].groupby("n")["gap_dsatur"].std()
        ax.plot(g.index, g, label=algo.upper(),
                color=ALGO_COLORS[algo], marker=ALGO_MARKERS[algo])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Standard Deviation of Chromatic Gap", "n", "Std dev of gap vs. DSATUR")
    _save(fig, "std_dev_gap.png", out_dir)


# ── Figure 6: density_sweep ───────────────────────────────────────────────────

def plot_density_sweep(
    out_dir: Path,
    ps: list[float] | None = None,
    reps: int = 10,
) -> None:
    """density_sweep.png: mean gap_dsatur vs edge density at fixed n=75.

    Runs the 4 metaheuristics directly (does not read from CSV).
    """
    if ps is None:
        ps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"Running density sweep: n=75  ps={ps}  reps={reps}")
    records: list[dict] = []
    for p in ps:
        print(f"  p={p:.1f} ...", end="", flush=True)
        for rep in range(reps):
            G = make_random_graph(75, p, seed=rep)
            dsatur_k = len(set(dsatur(G).values()))
            for name, run_fn, pkey in _RUNNERS:
                try:
                    result = run_fn(G, 75, _ALGO_PARAMS[pkey], seed=rep)  # type: ignore[operator]
                    records.append({"algo": name, "p": p, "gap": result.k_used - dsatur_k})
                except Exception:
                    pass
        print()
    sweep = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in META_ALGOS:
        g = sweep[sweep["algo"] == algo].groupby("p")["gap"].mean()
        ax.plot(g.index, g, label=algo.upper(),
                color=ALGO_COLORS[algo], marker=ALGO_MARKERS[algo])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Effect of Graph Density (n=75)", "Edge density p", "Mean gap vs. DSATUR")
    _save(fig, "density_sweep.png", out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all 6 figures. Convergence and density sweep are skippable."""
    p = argparse.ArgumentParser(description="Generate result figures.")
    p.add_argument("--csv", type=Path, default=None, help="Path to benchmark CSV")
    p.add_argument("--skip-convergence", action="store_true",
                   help="Skip the convergence experiment (saves ~15s)")
    p.add_argument("--skip-density", action="store_true",
                   help="Skip the density sweep (saves several minutes)")
    args = p.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv if args.csv else _latest_csv(BENCHMARK_DIR)
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)

    plot_avg_gap_vs_n(df, FIGURES_DIR)
    plot_gap_vs_bf_small_n(df, FIGURES_DIR)
    plot_avg_runtime(df, FIGURES_DIR)
    plot_std_dev_gap(df, FIGURES_DIR)
    if not args.skip_convergence:
        plot_convergence_n100(FIGURES_DIR)
    if not args.skip_density:
        plot_density_sweep(FIGURES_DIR)


if __name__ == "__main__":
    main()
