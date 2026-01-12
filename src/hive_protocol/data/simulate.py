"""Generate synthetic time series for testing Kalman filters.

This module provides utilities for simulating state-space model data,
enabling reproducible testing and demonstration of inference algorithms.

The generated data follows the linear Gaussian state-space model:
- State: x_t = x_{t-1} + w_t,  w_t ~ N(0, process_noise²)
- Observation: y_t = x_t + v_t,  v_t ~ N(0, measurement_noise²)

Example:
    >>> from hive_protocol.data import simulate_noisy_trajectory
    >>> states, observations = simulate_noisy_trajectory(
    ...     n_steps=100,
    ...     process_noise=0.1,
    ...     measurement_noise=0.5,
    ... )
    >>> print(f"True state range: [{states.min():.2f}, {states.max():.2f}]")
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray


def simulate_noisy_trajectory(
    n_steps: int = 100,
    process_noise: float = 0.1,
    measurement_noise: float = 0.5,
    initial_state: float = 0.0,
    seed: int | None = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate hidden states and noisy observations.

    Generates a random walk trajectory (hidden states) and corresponding
    noisy measurements. This is the canonical test case for Kalman filters.

    Args:
        n_steps: Number of time steps to simulate.
        process_noise: Standard deviation of state transition noise.
            Controls how much the hidden state varies between timesteps.
        measurement_noise: Standard deviation of observation noise.
            Controls how noisy the observations are relative to true state.
        initial_state: Starting value for the hidden state trajectory.
        seed: Random seed for reproducibility. None for random behavior.

    Returns:
        A tuple of (states, observations) where:
        - states: True hidden state trajectory (n_steps,)
        - observations: Noisy measurements (n_steps,)

    Raises:
        ValueError: If n_steps < 1 or noise parameters are negative.

    Example:
        >>> states, obs = simulate_noisy_trajectory(n_steps=50, seed=123)
        >>> # Observations should be noisier than states
        >>> assert np.std(obs - states) > 0
    """
    # Input validation
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if process_noise < 0:
        raise ValueError(f"process_noise must be >= 0, got {process_noise}")
    if measurement_noise < 0:
        raise ValueError(f"measurement_noise must be >= 0, got {measurement_noise}")

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate process noise (state transitions)
    process_noise_samples = rng.normal(0, process_noise, size=n_steps)

    # Generate measurement noise
    measurement_noise_samples = rng.normal(0, measurement_noise, size=n_steps)

    # Simulate random walk (cumulative sum of process noise)
    # First element starts at initial_state
    states = np.zeros(n_steps, dtype=np.float64)
    states[0] = initial_state + process_noise_samples[0]
    for t in range(1, n_steps):
        states[t] = states[t - 1] + process_noise_samples[t]

    # Generate observations (states + measurement noise)
    observations = states + measurement_noise_samples

    return states, observations


def simulate_multiple_trajectories(
    n_trajectories: int = 10,
    n_steps: int = 100,
    process_noise: float = 0.1,
    measurement_noise: float = 0.5,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Simulate multiple trajectories and return as a Polars DataFrame.

    Useful for batch experiments and parameter sweeps. Returns data
    in tidy format suitable for visualization and analysis.

    Args:
        n_trajectories: Number of independent trajectories to simulate.
        n_steps: Number of time steps per trajectory.
        process_noise: Standard deviation of state transition noise.
        measurement_noise: Standard deviation of observation noise.
        seed: Base random seed. Each trajectory uses seed + trajectory_id.

    Returns:
        Polars DataFrame with columns:
        - trajectory_id: Integer identifying the trajectory (0 to n-1)
        - timestep: Time index within trajectory (0 to n_steps-1)
        - true_state: Hidden state value
        - observation: Noisy measurement
        - process_noise: Process noise parameter used
        - measurement_noise: Measurement noise parameter used

    Example:
        >>> df = simulate_multiple_trajectories(n_trajectories=5, n_steps=50)
        >>> print(df.group_by("trajectory_id").agg(pl.col("observation").std()))
    """
    records = []

    for traj_id in range(n_trajectories):
        # Use different seed for each trajectory for independence
        traj_seed = seed + traj_id if seed is not None else None

        states, observations = simulate_noisy_trajectory(
            n_steps=n_steps,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            seed=traj_seed,
        )

        for t in range(n_steps):
            records.append(
                {
                    "trajectory_id": traj_id,
                    "timestep": t,
                    "true_state": states[t],
                    "observation": observations[t],
                    "process_noise": process_noise,
                    "measurement_noise": measurement_noise,
                }
            )

    return pl.DataFrame(records)


def simulate_parameter_sweep(
    process_noise_values: list[float],
    measurement_noise_values: list[float],
    n_steps: int = 100,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Simulate trajectories across a grid of noise parameters.

    Creates one trajectory for each combination of process and
    measurement noise values. Useful for studying how noise
    levels affect inference quality.

    Args:
        process_noise_values: List of process noise values to test.
        measurement_noise_values: List of measurement noise values to test.
        n_steps: Number of time steps per trajectory.
        seed: Base random seed for reproducibility.

    Returns:
        Polars DataFrame in tidy format with all parameter combinations.

    Example:
        >>> df = simulate_parameter_sweep(
        ...     process_noise_values=[0.1, 0.5, 1.0],
        ...     measurement_noise_values=[0.1, 0.5, 1.0],
        ... )
        >>> print(df.group_by(["process_noise", "measurement_noise"]).len())
    """
    records = []
    experiment_id = 0

    for proc_noise in process_noise_values:
        for meas_noise in measurement_noise_values:
            # Different seed for each parameter combination
            exp_seed = seed + experiment_id if seed is not None else None

            states, observations = simulate_noisy_trajectory(
                n_steps=n_steps,
                process_noise=proc_noise,
                measurement_noise=meas_noise,
                seed=exp_seed,
            )

            for t in range(n_steps):
                records.append(
                    {
                        "experiment_id": experiment_id,
                        "timestep": t,
                        "true_state": states[t],
                        "observation": observations[t],
                        "process_noise": proc_noise,
                        "measurement_noise": meas_noise,
                    }
                )

            experiment_id += 1

    return pl.DataFrame(records)
