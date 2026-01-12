"""Hive-Protocol: The definitive protocol for Python in computational biology.

A template repository demonstrating modern Python standards for computational
biology research at CBG-ETH Zurich. Built for the D-BSSE Wise Symposium workshop
(January 21-23, 2026).

Example:
    >>> from hive_protocol.data import simulate_noisy_trajectory
    >>> from hive_protocol.inference import fit_kalman_filter
    >>> states, observations = simulate_noisy_trajectory(n_steps=100)
    >>> model, trace = fit_kalman_filter(observations)
"""

from hive_protocol._version import __version__

__all__ = ["__version__"]
