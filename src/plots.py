"""Generate all result figures from the benchmark CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless cluster nodes
import matplotlib.pyplot as plt
import pandas as pd

from algorithms import aco, ga, sa, hybrid_ga_sa, hybrid_aco_sa
from config import ALGO_PARAMS as _ALGO_PARAMS
from graph_utils import dsatur, make_random_graph

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
BENCHMARK_DIR = Path(__file__).parent.parent / "results" / "benchmark"
DPI = 300

# ── Consistent style per algorithm across all figures ─────────────────────────
ALGO_COLORS: dict[str, str] = {
    "ga": "#1f77b4",           # blue
    "gae": "#ff7f0e",          # orange
    "aco": "#2ca02c",          # green
    "sa": "#d62728",           # red
    "hybrid_ga_sa": "#9467bd", # purple
    "hybrid_aco_sa": "#e377c2", # pink - distinct from others
    "dsatur": "#8c564b",       # brown
    "bf": "#17becf",           # cyan
}
ALGO_MARKERS: dict[str, str] = {
    "ga": "o",
    "gae": "s",
    "aco": "^",
    "sa": "D",
    "hybrid_ga_sa": "*",       # star marker
    "hybrid_aco_sa": "P",      # plus (filled) marker - stands out
    "dsatur": "x",
    "bf": "+",
}
# Metaheuristics only (exclude baselines)
META_ALGOS = ["ga", "gae", "aco", "sa", "hybrid_ga_sa", "hybrid_aco_sa"]

_RUNNERS = [
    ("ga", ga.run, "ga"),
    ("gae", ga.run, "gae"),
    ("aco", aco.run, "aco"),
    ("sa", sa.run, "sa"),
    ("hybrid_ga_sa", hybrid_ga_sa.run, "hybrid_ga_sa"),
    ("hybrid_aco_sa", hybrid_aco_sa.run_aco_sa, "hybrid_aco_sa"),
]


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
        if g.mean().empty:
            continue
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
    if sub.empty:
        print("Warning: No gap_bf data found; skipping plot_gap_vs_bf_small_n")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in META_ALGOS:
        g = sub[sub["algo"] == algo].groupby("n")["gap_bf"]
        if g.mean().empty:
            continue
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
    # Include all algorithms: baselines + metaheuristics + hybrids
    all_algos = ["dsatur", "bf"] + META_ALGOS
    for algo in all_algos:
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
        print(f"  Running {name} for convergence...", end="", flush=True)
        try:
            result = run_fn(G, 100, _ALGO_PARAMS[pkey], seed=0)  # type: ignore[operator]
            histories[name] = result.fitness_history
            print(f" done ({len(result.fitness_history)} points)")
        except Exception as e:
            print(f" failed: {e}")
            histories[name] = []
    return histories


def plot_convergence_n100(out_dir: Path) -> None:
    """convergence_n100.png: fitness history per algo on G(n=100, p=0.5, seed=0).

    The x-axis is normalised to 0–100 % of each algorithm's own budget so that
    convergence *shape* can be compared fairly.  Raw iteration counts differ:
    GA records one point per generation (200 total), ACO one per outer iteration
    (300), SA one per n_step-move block (200), Hybrid one per generation (150).
    Absolute wall-clock speed is shown separately in avg_runtime.png.
    """
    print("Running convergence experiment on G(n=100, p=0.5, seed=0) ...")
    histories = run_convergence_experiment()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter out empty histories
    for algo, hist in histories.items():
        if not hist:
            continue
        pct = [100.0 * i / (len(hist) - 1) for i in range(len(hist))]
        ax.plot(pct, hist, label=algo.upper(), 
                color=ALGO_COLORS.get(algo, "gray"), 
                marker=ALGO_MARKERS.get(algo, ""),
                markersize=3, markevery=len(hist)//10, alpha=0.85)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Convergence at n=100 (normalised budget)",
           "% of algorithm budget", "Best Fitness F")
    _save(fig, "convergence_n100.png", out_dir)


# ── Figure 5: std_dev_gap ─────────────────────────────────────────────────────

def plot_std_dev_gap(df: pd.DataFrame, out_dir: Path) -> None:
    """std_dev_gap.png: std dev of gap_dsatur per algo vs n."""
    sub = df[df["algo"].isin(META_ALGOS)].copy()
    sub["gap_dsatur"] = pd.to_numeric(sub["gap_dsatur"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in META_ALGOS:
        g = sub[sub["algo"] == algo].groupby("n")["gap_dsatur"].std()
        if g.empty:
            continue
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

    Runs the metaheuristics directly (does not read from CSV).
    Includes both hybrid algorithms.
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
    
    if sweep.empty:
        print("Warning: No data collected for density sweep")
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in META_ALGOS:
        g = sweep[sweep["algo"] == algo].groupby("p")["gap"].mean()
        if g.empty:
            continue
        ax.plot(g.index, g, label=algo.upper(),
                color=ALGO_COLORS.get(algo, "gray"), 
                marker=ALGO_MARKERS.get(algo, "o"))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Effect of Graph Density (n=75)", "Edge density p", "Mean gap vs. DSATUR")
    _save(fig, "density_sweep.png", out_dir)


# ── Figure 7: hybrid_comparison ───────────────────────────────────────────────

def plot_hybrid_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """hybrid_comparison.png: Direct comparison of both hybrids vs all other algorithms.
    
    Shows the relative performance of both hybrid approaches.
    """
    sub = df[df["algo"].isin(META_ALGOS)].copy()
    sub["gap_dsatur"] = pd.to_numeric(sub["gap_dsatur"], errors="coerce")
    
    # Get hybrid data
    hybrid_ga_sa_data = sub[sub["algo"] == "hybrid_ga_sa"].groupby("n")["gap_dsatur"].mean()
    hybrid_aco_sa_data = sub[sub["algo"] == "hybrid_aco_sa"].groupby("n")["gap_dsatur"].mean()
    
    if hybrid_ga_sa_data.empty and hybrid_aco_sa_data.empty:
        print("Warning: No hybrid data found; skipping plot_hybrid_comparison")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot each algorithm's gap as a line
    for algo in META_ALGOS:
        if algo in ["hybrid_ga_sa", "hybrid_aco_sa"]:
            continue
        g = sub[sub["algo"] == algo].groupby("n")["gap_dsatur"].mean()
        if g.empty:
            continue
        # Plot as thin, semi-transparent lines for comparison
        ax.plot(g.index, g, label=algo.upper(),
                color=ALGO_COLORS[algo], marker=ALGO_MARKERS[algo],
                alpha=0.4, linestyle="--", linewidth=1.5)
    
    # Plot hybrids as bold, prominent lines
    if not hybrid_ga_sa_data.empty:
        ax.plot(hybrid_ga_sa_data.index, hybrid_ga_sa_data, 
                label="HYBRID_GA_SA",
                color=ALGO_COLORS["hybrid_ga_sa"], 
                marker=ALGO_MARKERS["hybrid_ga_sa"],
                linewidth=3, markersize=10)
    
    if not hybrid_aco_sa_data.empty:
        ax.plot(hybrid_aco_sa_data.index, hybrid_aco_sa_data, 
                label="HYBRID_ACO_SA",
                color=ALGO_COLORS["hybrid_aco_sa"], 
                marker=ALGO_MARKERS["hybrid_aco_sa"],
                linewidth=3, markersize=10)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _style(ax, "Hybrid Algorithms Performance Comparison", 
           "n", "Mean gap vs. DSATUR")
    _save(fig, "hybrid_comparison.png", out_dir)


# ── Figure 8: hybrid_improvement ─────────────────────────────────────────────

def plot_hybrid_improvement(df: pd.DataFrame, out_dir: Path) -> None:
    """hybrid_improvement.png: Percentage improvement of both hybrids over others."""
    sub = df[df["algo"].isin(META_ALGOS)].copy()
    sub["gap_dsatur"] = pd.to_numeric(sub["gap_dsatur"], errors="coerce")
    
    # Get hybrid data
    hybrid_data = {}
    for h in ["hybrid_ga_sa", "hybrid_aco_sa"]:
        hybrid_data[h] = sub[sub["algo"] == h].groupby("n")["gap_dsatur"].mean()
        if hybrid_data[h].empty:
            hybrid_data[h] = None
    
    if all(v is None for v in hybrid_data.values()):
        print("Warning: No hybrid data found; skipping plot_hybrid_improvement")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # For each hybrid, calculate improvement over each algorithm
    for hybrid_name, hybrid_vals in hybrid_data.items():
        if hybrid_vals is None:
            continue
            
        for algo in META_ALGOS:
            if algo == hybrid_name:
                continue
            g = sub[sub["algo"] == algo].groupby("n")["gap_dsatur"].mean()
            if g.empty:
                continue
            
            # Calculate improvement
            common_n = g.index.intersection(hybrid_vals.index)
            if len(common_n) == 0:
                continue
            
            improvement = []
            for n in common_n:
                other_gap = g.loc[n]
                hybrid_gap = hybrid_vals.loc[n]
                if other_gap > 0:
                    imp = (other_gap - hybrid_gap) / other_gap * 100
                else:
                    imp = 0.0 if hybrid_gap == 0 else 100.0
                improvement.append(imp)
            
            # Use dashed/solid to distinguish hybrids
            linestyle = "-" if hybrid_name == "hybrid_ga_sa" else "--"
            label = f"{hybrid_name.upper()} vs {algo.upper()}"
            ax.plot(common_n, improvement, label=label,
                    color=ALGO_COLORS[algo], 
                    marker=ALGO_MARKERS[algo],
                    linestyle=linestyle,
                    alpha=0.7)
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _style(ax, "Hybrid Algorithms: Percentage Improvement Over Others",
           "n", "Improvement (%)")
    _save(fig, "hybrid_improvement.png", out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all 8 figures. Convergence and density sweep are skippable."""
    p = argparse.ArgumentParser(description="Generate result figures.")
    p.add_argument("--csv", type=Path, default=None, help="Path to benchmark CSV")
    p.add_argument("--skip-convergence", action="store_true",
                   help="Skip the convergence experiment (saves ~15s)")
    p.add_argument("--skip-density", action="store_true",
                   help="Skip the density sweep (saves several minutes)")
    p.add_argument("--skip-hybrid-plots", action="store_true",
                   help="Skip hybrid-specific comparison plots")
    args = p.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv if args.csv else _latest_csv(BENCHMARK_DIR)
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)

    # Standard plots
    plot_avg_gap_vs_n(df, FIGURES_DIR)
    plot_gap_vs_bf_small_n(df, FIGURES_DIR)
    plot_avg_runtime(df, FIGURES_DIR)
    plot_std_dev_gap(df, FIGURES_DIR)
    
    # Optional experiments
    if not args.skip_convergence:
        plot_convergence_n100(FIGURES_DIR)
    if not args.skip_density:
        plot_density_sweep(FIGURES_DIR)
    
    # Hybrid-specific plots (new)
    if not args.skip_hybrid_plots:
        plot_hybrid_comparison(df, FIGURES_DIR)
        plot_hybrid_improvement(df, FIGURES_DIR)


if __name__ == "__main__":
    main()