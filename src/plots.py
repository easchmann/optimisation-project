"""Generate all result figures from benchmark and tuning CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import pandas as pd

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
BENCHMARK_DIR = Path(__file__).parent.parent / "results" / "benchmark"
DPI = 300


def _latest_csv(directory: Path) -> Path:
    """Return the most recently modified CSV in a directory."""
    csvs = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csvs[-1]


def plot_gap_vs_n(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot chromatic gap vs graph size n, one line per algorithm.

    Args:
        df: Benchmark results dataframe.
        out_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, p in zip(axes, [0.3, 0.5, 0.7]):
        sub = df[df["p"] == p]
        for algo, grp in sub.groupby("algo"):
            mean_gap = grp.groupby("n")["gap_dsatur"].mean()
            ax.plot(mean_gap.index, mean_gap.values, marker="o", label=algo)
        ax.set_title(f"p = {p}")
        ax.set_xlabel("n")
        ax.set_ylabel("Chromatic gap (vs DSATUR)")
        ax.legend()
    fig.tight_layout()
    out = out_dir / "gap_vs_n.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


def plot_runtime_vs_n(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot mean runtime vs n, one line per algorithm.

    Args:
        df: Benchmark results dataframe.
        out_dir: Directory to save figures.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, grp in df.groupby("algo"):
        mean_rt = grp.groupby("n")["runtime_s"].mean()
        ax.plot(mean_rt.index, mean_rt.values, marker="o", label=algo)
    ax.set_xlabel("n")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Mean runtime vs graph size")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "runtime_vs_n.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


def plot_convergence(history: list[float], algo: str, out_dir: Path) -> None:
    """Plot fitness convergence curve for a single run.

    Args:
        history: List of best fitness values per iteration.
        algo: Algorithm name (used in title and filename).
        out_dir: Directory to save figures.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness F")
    ax.set_title(f"{algo} convergence")
    fig.tight_layout()
    out = out_dir / f"convergence_{algo.lower()}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    """Generate all figures from the latest benchmark CSV."""
    parser = argparse.ArgumentParser(description="Generate result figures.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to benchmark CSV")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv if args.csv else _latest_csv(BENCHMARK_DIR)
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)

    plot_gap_vs_n(df, FIGURES_DIR)
    plot_runtime_vs_n(df, FIGURES_DIR)


if __name__ == "__main__":
    main()
