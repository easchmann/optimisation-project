# Graph Coloring via Metaheuristic Optimization
**OMfE 227-0707-00L · ETH Zürich · Spring 2026**

This project compares four metaheuristic algorithms — Genetic Algorithm (GA), GA with Elitism (GAE), Ant Colony Optimization (ACO), and Simulated Annealing (SA) — against two baselines (DSATUR greedy heuristic and Brute Force) on the graph coloring problem. Each algorithm is evaluated across random Erdős–Rényi graphs of increasing size (n = 20 … 200) using the chromatic gap (colours used − χ(G)) as the primary metric. Lower is better; 0 is optimal.

## Installation

```bash
pip install networkx numpy matplotlib scipy deap pandas
```
## Reproducing results

Run the three steps in order:

```bash
# 1. Tune hyperparameters (one algorithm at a time; writes CSV to results/tuning/)
python src/tune.py --algo sa --reps 10 --ns 40 50 60
python src/tune.py --algo aco --reps 10 --ns 40 50 60
python src/tune.py --algo ga  --reps 10 --ns 40 50 60

# 2. Full benchmark sweep (writes CSV to results/benchmark/)
python src/benchmark.py --ns 20 30 40 50 75 100 150 200 --reps 20 --ps 0.5

# 3. Generate all figures (reads latest benchmark CSV; writes PNGs to results/figures/)
python src/plots.py
# Add --skip-density to skip the slow density sweep (~10 min)
# Add --skip-convergence to skip the convergence experiment (~15 s)
```

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
└── figures/
    ├── avg_gap_vs_n.png        # mean chromatic gap ± std vs n
    ├── gap_vs_bf_small_n.png   # gap vs optimal for n ≤ 15
    ├── avg_runtime.png         # runtime (log scale) vs n
    ├── convergence_n100.png    # fitness history at n=100
    ├── std_dev_gap.png         # variability of gap vs n
    └── density_sweep.png       # gap vs edge density at n=75
```
