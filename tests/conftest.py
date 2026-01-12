"""Pytest configuration and shared fixtures.

This module provides reusable test fixtures for the hive-protocol
test suite. Fixtures ensure consistent test data across test modules.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_observations() -> np.ndarray:
    """Generate simple test observations.

    Returns a short time series suitable for fast test execution.
    The data follows a noisy random walk pattern.
    """
    rng = np.random.default_rng(42)
    n_steps = 30
    states = np.cumsum(rng.normal(0, 0.1, n_steps))
    observations = states + rng.normal(0, 0.3, n_steps)
    return observations


@pytest.fixture
def long_observations() -> np.ndarray:
    """Generate longer test observations for integration tests.

    Returns a longer time series for more thorough testing.
    """
    rng = np.random.default_rng(123)
    n_steps = 100
    states = np.cumsum(rng.normal(0, 0.2, n_steps))
    observations = states + rng.normal(0, 0.5, n_steps)
    return observations


@pytest.fixture
def known_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory with known true states.

    Returns both the true hidden states and noisy observations,
    useful for testing filter accuracy.
    """
    rng = np.random.default_rng(456)
    n_steps = 50
    true_states = np.cumsum(rng.normal(0, 0.15, n_steps))
    observations = true_states + rng.normal(0, 0.4, n_steps)
    return true_states, observations


@pytest.fixture
def constant_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Generate a constant trajectory (no process noise).

    Useful for testing edge cases where the true state doesn't change.
    """
    n_steps = 30
    true_states = np.full(n_steps, 5.0)
    rng = np.random.default_rng(789)
    observations = true_states + rng.normal(0, 0.5, n_steps)
    return true_states, observations
