"""Canonical algorithm parameter tables and benchmark settings.

All hyperparameters here reflect the tuned defaults documented in CLAUDE.md.
Import from this module instead of copying the dicts into each script.
"""

from __future__ import annotations

# Tuned defaults (do not change without re-running tune.py).
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
        "n_ants": 50, "alpha": 1.0, "beta": 3.0, "rho": 0.2,
        "Q": 1.0, "tau_min": 0.01, "n_iter": 300,
    },
    "sa": {
        "T0": 100.0, "gamma": 0.995, "n_step": None, "n_stall": 500, "n_max": None,
    },
}
