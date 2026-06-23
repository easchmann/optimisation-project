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
    "hybrid_ga_sa": {
        "n_pop": 80,
        "p_cx": 0.8,
        "p_mut": 0.3,
        "p_ind": 0.02,
        "n_gen": 120,
        "n_elite": 2,
        "t_size": 5,
        "elitism": True,
        "sa_steps": None,  # Will become 15*n at runtime
        "sa_T0": 6.0,
        "sa_gamma": 0.95,
        "sa_refine_fraction": 0.12,
        #"n_pop": 80,
        #"p_cx": 0.8,
        #"p_mut": 0.3,
        #"p_ind": 0.02,
        #"n_gen": 80,
        #"n_elite": 2,
        #"t_size": 5,
        #"elitism": True,
        #"sa_steps": 20,  # Fixed, not 50*n
        #"sa_T0": 2.0,
        #"sa_gamma": 0.95,
        #"sa_refine_fraction": 0.05, 
    },
    "hybrid_aco_sa": {
        "aco_iter": 100,
        "aco_ants": 50,
        "aco_alpha": 2.0,
        "aco_beta": 5.0,
        "aco_rho": 0.2,
        "aco_Q": 1.0,
        "sa_steps": None,      # 100*n at runtime
        "sa_T0": 20.0,
        "sa_gamma": 0.99,
        "sa_restarts": 3,
    },
}
