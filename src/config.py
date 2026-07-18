"""Canonical algorithm parameter tables and benchmark settings.

All hyperparameters here reflect the tuned defaults documented in CLAUDE.md.
Import from this module instead of copying the dicts into each script.
"""

from __future__ import annotations

# Tuned defaults (do not change without re-running tune.py).
ALGO_PARAMS: dict[str, dict] = {
    "ga": {"n_pop": 100, "p_cx": 0.9, "p_mut": 0.5, "p_ind": 0.05, "n_gen": 200, "n_elite": 3, "t_size": 5, "elitism": False},
    "gae": {"n_pop": 100, "p_cx": 1, "p_mut": 0.5, "p_ind": 0.05, "n_gen": 200, "n_elite": 3, "t_size": 5, "elitism": True},
    "aco": {"n_ants": 200, "alpha": 2.0, "beta": 5.0, "rho": 0.1, "Q": 1.0, "tau_min": 0.01, "n_iter": 300},
    "sa": {"T0": 100, "gamma": 0.999, "n_step": None, "n_stall": 100, "n_max": None},
    "hybrid_ga_sa": {"n_pop": 80, "p_cx": 0.6, "p_mut": 0.4, "p_ind": 0.1, "n_gen": 120, "n_elite": 1, "t_size": 4, "elitism": True, "sa_steps": None, "sa_steps_factor": 100, "sa_T0": 1, "sa_gamma": 0.9, "sa_refine_fraction": 0.2},
    "hybrid_aco_sa": {"aco_iter": 100, "aco_ants": 20, "aco_alpha": 1, "aco_beta": 3, "aco_rho": 0.2, "aco_Q": 1.0, "sa_steps": None, "sa_steps_factor": 150, "sa_T0": 10, "sa_gamma": 0.9, "sa_restarts": 5},
}
