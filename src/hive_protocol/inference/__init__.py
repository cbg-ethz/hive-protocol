"""Inference module for Bayesian state estimation.

This module provides implementations of state-space models using PyMC,
focusing on Kalman filters as a teaching example for Bayesian inference
in time series analysis.

Example:
    >>> from hive_protocol.inference import fit_kalman_filter
    >>> from hive_protocol.data import simulate_noisy_trajectory
    >>> states, obs = simulate_noisy_trajectory(n_steps=50)
    >>> model, trace = fit_kalman_filter(obs)
"""

from hive_protocol.inference.diagnostics import (
    check_convergence,
    compute_prediction_errors,
    generate_diagnostic_report,
    summarize_filter_performance,
)
from hive_protocol.inference.kalman import (
    extract_filtered_states,
    fit_kalman_filter,
    get_noise_estimates,
)

__all__ = [
    "check_convergence",
    "compute_prediction_errors",
    "extract_filtered_states",
    "fit_kalman_filter",
    "generate_diagnostic_report",
    "get_noise_estimates",
    "summarize_filter_performance",
]
