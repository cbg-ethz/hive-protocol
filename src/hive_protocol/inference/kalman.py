"""Kalman filter implementation using PyMC state-space models.

This module demonstrates Bayesian state-space modeling using PyMC's
state-space module from pymc-extras. The Kalman filter is a fundamental
algorithm for tracking hidden states through noisy observations.

Theory:
    The linear Gaussian state-space model consists of:
    - State equation: x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
    - Observation equation: y_t = H * x_t + v_t,  v_t ~ N(0, R)

    Where:
    - x_t: hidden state at time t
    - y_t: observation at time t
    - A: state transition matrix
    - H: observation matrix
    - Q: process noise covariance
    - R: measurement noise covariance

Example:
    >>> import numpy as np
    >>> from hive_protocol.inference import fit_kalman_filter
    >>> observations = np.random.randn(50)  # noisy measurements
    >>> model, trace = fit_kalman_filter(observations)
"""

from typing import Any

import arviz as az
import numpy as np
import pymc as pm
from numpy.typing import NDArray


def fit_kalman_filter(
    observations: NDArray[np.float64],
    process_variance_prior: float = 1.0,
    measurement_variance_prior: float = 1.0,
    n_samples: int = 1000,
    n_tune: int = 500,
    random_seed: int | None = None,
) -> tuple[pm.Model, az.InferenceData]:
    """Fit a Kalman filter to time series data using PyMC.

    This implements a simple random walk state-space model where the
    hidden state evolves as a random walk and observations are noisy
    measurements of that state.

    Args:
        observations: Time series measurements (1D array of floats).
        process_variance_prior: Prior scale for state transition noise.
            Larger values allow more state variability between timesteps.
        measurement_variance_prior: Prior scale for observation noise.
            Larger values indicate less trust in individual measurements.
        n_samples: Number of posterior samples to draw after tuning.
        n_tune: Number of tuning samples for NUTS adaptation.
        random_seed: Random seed for reproducibility.

    Returns:
        A tuple of (PyMC Model, ArviZ InferenceData) containing the
        fitted model and posterior samples.

    Raises:
        ValueError: If observations is empty or contains NaN values.

    Example:
        >>> import numpy as np
        >>> obs = np.cumsum(np.random.randn(50)) + np.random.randn(50) * 0.5
        >>> model, trace = fit_kalman_filter(obs, n_samples=500)
        >>> print(az.summary(trace, var_names=["process_noise", "measurement_noise"]))
    """
    # Input validation
    observations = np.asarray(observations, dtype=np.float64)
    if observations.size == 0:
        raise ValueError("observations array cannot be empty")
    if np.any(np.isnan(observations)):
        raise ValueError("observations cannot contain NaN values")

    n_timesteps = len(observations)

    # Build the PyMC model
    # We use a hierarchical approach where noise variances are learned
    with pm.Model() as model:
        # --- Priors for noise parameters ---
        # HalfNormal priors ensure positive variances
        # The prior scale controls how much noise we expect a priori
        process_noise = pm.HalfNormal(
            "process_noise",
            sigma=process_variance_prior,
        )
        measurement_noise = pm.HalfNormal(
            "measurement_noise",
            sigma=measurement_variance_prior,
        )

        # --- Initial state prior ---
        # Center on first observation with wide uncertainty
        initial_state = pm.Normal(
            "initial_state",
            mu=observations[0],
            sigma=2.0,
        )

        # --- State-space model using GaussianRandomWalk ---
        # This is equivalent to: x_t = x_{t-1} + w_t, w_t ~ N(0, sigma)
        # GaussianRandomWalk efficiently represents the random walk prior
        states = pm.GaussianRandomWalk(
            "states",
            sigma=process_noise,
            init_dist=pm.Normal.dist(mu=initial_state, sigma=0.01),
            steps=n_timesteps - 1,
            shape=n_timesteps,
        )

        # --- Observation likelihood ---
        # y_t ~ N(x_t, measurement_noise)
        pm.Normal(
            "likelihood",
            mu=states,
            sigma=measurement_noise,
            observed=observations,
        )

        # --- Posterior sampling ---
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, trace


def extract_filtered_states(
    trace: az.InferenceData,
    credible_interval: float = 0.94,
) -> dict[str, NDArray[np.float64]]:
    """Extract filtered state estimates from posterior samples.

    Computes point estimates and credible intervals for the hidden
    states from the posterior distribution.

    Args:
        trace: ArviZ InferenceData from fit_kalman_filter.
        credible_interval: Width of the credible interval (0-1).

    Returns:
        Dictionary with keys:
        - 'mean': Posterior mean of states
        - 'median': Posterior median of states
        - 'lower': Lower bound of credible interval
        - 'upper': Upper bound of credible interval

    Example:
        >>> model, trace = fit_kalman_filter(observations)
        >>> states = extract_filtered_states(trace)
        >>> plt.fill_between(range(len(states['mean'])),
        ...                  states['lower'], states['upper'], alpha=0.3)
        >>> plt.plot(states['mean'], label='Filtered state')
    """
    # Get the states posterior samples
    states_samples = trace.posterior["states"].values

    # Flatten chains: (n_chains, n_samples, n_timesteps) -> (n_total, n_timesteps)
    n_chains, n_samples, n_timesteps = states_samples.shape
    states_flat = states_samples.reshape(-1, n_timesteps)

    # Compute summary statistics
    alpha = 1 - credible_interval
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    return {
        "mean": np.mean(states_flat, axis=0),
        "median": np.median(states_flat, axis=0),
        "lower": np.quantile(states_flat, lower_q, axis=0),
        "upper": np.quantile(states_flat, upper_q, axis=0),
    }


def get_noise_estimates(
    trace: az.InferenceData,
) -> dict[str, dict[str, float]]:
    """Extract noise parameter estimates from posterior.

    Args:
        trace: ArviZ InferenceData from fit_kalman_filter.

    Returns:
        Dictionary with posterior summaries for process_noise
        and measurement_noise parameters.

    Example:
        >>> model, trace = fit_kalman_filter(observations)
        >>> estimates = get_noise_estimates(trace)
        >>> print(f"Process noise: {estimates['process_noise']['mean']:.3f}")
        >>> print(f"Measurement noise: {estimates['measurement_noise']['mean']:.3f}")
    """
    summary: Any = az.summary(
        trace,
        var_names=["process_noise", "measurement_noise"],
    )

    return {
        "process_noise": {
            "mean": float(summary.loc["process_noise", "mean"]),
            "sd": float(summary.loc["process_noise", "sd"]),
            "hdi_3%": float(summary.loc["process_noise", "hdi_3%"]),
            "hdi_97%": float(summary.loc["process_noise", "hdi_97%"]),
        },
        "measurement_noise": {
            "mean": float(summary.loc["measurement_noise", "mean"]),
            "sd": float(summary.loc["measurement_noise", "sd"]),
            "hdi_3%": float(summary.loc["measurement_noise", "hdi_3%"]),
            "hdi_97%": float(summary.loc["measurement_noise", "hdi_97%"]),
        },
    }
