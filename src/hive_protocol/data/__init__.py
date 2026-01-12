"""Data generation module for synthetic time series.

This module provides utilities for generating simulated time series data,
useful for testing inference algorithms and demonstrating Kalman filtering.

Example:
    >>> from hive_protocol.data import simulate_noisy_trajectory
    >>> states, observations = simulate_noisy_trajectory(n_steps=100)
    >>> print(f"Generated {len(observations)} observations")
"""

from hive_protocol.data.simulate import simulate_noisy_trajectory

__all__ = ["simulate_noisy_trajectory"]
