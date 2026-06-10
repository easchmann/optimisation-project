"""Canonical algorithm parameter tables and benchmark settings.

All hyperparameters here reflect the tuned defaults documented in CLAUDE.md.
Import from this module instead of copying the dicts into each script.
"""

from __future__ import annotations

# Tuned defaults (do not change without re-running tune.py).
ALGO_PARAMS: dict[str, dict] = {
    "ga": {"n_pop": 100, "p_cx": 0.9, "p_mut": 0.5, "p_ind": 0.05, "n_gen": 200, "n_elite": 3, "t_size": 5, "elitism": False},
    "gae": {"n_pop": 100, "p_cx": 0.7, "p_mut": 0.5, "p_ind": 0.01, "n_gen": 200, "n_elite": 3, "t_size": 5, "elitism": True},
    "aco": {"n_ants": 150, "alpha": 2.0, "beta": 5.0, "rho": 0.2, "Q": 1.0, "tau_min": 0.01, "n_iter": 300},
    "sa": {"T0": 10, "gamma": 0.999, "n_step": None, "n_stall": 100, "n_max": None},
}
