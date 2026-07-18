# Graph Coloring via Metaheuristic Optimization
**OMfE 227-0707-00L · ETH Zürich · Spring 2026**

This project compares six metaheuristic algorithms — Genetic Algorithm (GA), GA with Elitism (GAE), Ant Colony Optimization (ACO), Simulated Annealing (SA), and two hybrids (GA+SA, ACO+SA) — against two baselines (DSATUR greedy heuristic and Brute Force) on the graph coloring problem. Each algorithm is evaluated across random Erdős–Rényi graphs of increasing size (n = 20 … 200), plus a fixed set of DIMACS benchmark instances with known χ(G), using the chromatic gap (colours used − χ(G)) as the primary metric. Lower is better; 0 is optimal.

## Installation

```bash
pip install networkx numpy matplotlib scipy deap pandas
```
## Reproducing results

Run the four steps in order:

```bash
# 1. Tune hyperparameters (one algorithm at a time; writes CSV to results/tuning/)
python src/tune.py --algo sa --reps 10 --ns 40 50 60
python src/tune.py --algo aco --reps 10 --ns 40 50 60
python src/tune.py --algo ga  --reps 10 --ns 40 50 60

# 2. Full benchmark sweep (writes CSV to results/benchmark/)
python src/benchmark.py --ns 20 30 40 50 75 100 150 200 --reps 20 --ps 0.5

# 3. DIMACS benchmark: fixed instances with a known true χ(G) (writes CSV to
#    results/dimacs/). hybrid_ga_sa/aco are slow at the larger instances
#    (myciel7, n=191) — try a subset first:
python src/dimacs_benchmark.py --instances myciel3 myciel4 --reps 3
# Full sweep (runs serially; can take a while for the larger instances):
python src/dimacs_benchmark.py --reps 10

# 4. Generate all figures (reads latest benchmark + DIMACS CSVs; writes PNGs to results/figures/)
python src/plots.py
# Add --skip-density to skip the slow density sweep (~10 min)
# Add --skip-convergence to skip the convergence experiment (~15 s)
# Add --skip-dimacs to skip the DIMACS figure if step 3 hasn't been run yet
```

## Running on the cluster

`cluster/run_all.sh` submits the full pipeline as SLURM array jobs. It's two
phases with a deliberate checkpoint in between, since applying tuning results
overwrites `src/config.py` and that shouldn't happen unattended:

```bash
# 1. Tuning sweep (132 tasks)
./cluster/run_all.sh tune
# ...when it finishes:
python cluster/apply_tuning.py          # review the diff
python cluster/apply_tuning.py --apply  # write it to src/config.py

# 2. Random-graph benchmark (24 tasks) + DIMACS benchmark (13 tasks) in
#    parallel, then merge + plots.py, all chained via SLURM job dependencies
./cluster/run_all.sh benchmark
```

See `cluster/*.sh` headers for per-script details (array sizes, time budgets,
what each merge step expects).

## Single experiment

```bash
# Run all algorithms on one graph and print a comparison table
python src/main.py --n 50 --p 0.5 --seed 42 --k_max 20 --algo all

# Run a specific algorithm only
python src/main.py --n 30 --p 0.4 --seed 7 --algo sa
```

## Tests

```bash
python -m pytest tests/ -v
```

## Output structure

```
results/
├── tuning/
│   └── {algo}_tuning_{timestamp}.csv   # gap per (varied_param, param_value, n, rep)
├── benchmark/
│   └── results_{timestamp}.csv         # k_used, violations, gap_dsatur, gap_bf, runtime_s
├── dimacs/
│   └── dimacs_results_{timestamp}.csv  # k_used, violations, gap_true, gap_dsatur, runtime_s
└── figures/
    ├── avg_gap_vs_n.png        # mean chromatic gap ± std vs n
    ├── gap_vs_bf_small_n.png   # gap vs optimal for n ≤ 15
    ├── avg_runtime.png         # runtime (log scale) vs n
    ├── convergence_n100.png    # fitness history at n=100
    ├── std_dev_gap.png         # variability of gap vs n
    ├── density_sweep.png       # gap vs edge density at n=75
    └── dimacs_gap.png          # gap vs true χ(G) per DIMACS instance
```
