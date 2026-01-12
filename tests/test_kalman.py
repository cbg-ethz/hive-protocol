"""Tests for Kalman filter inference module.

Tests cover:
- Input validation
- Model structure
- Output formats
- Integration with ArviZ
- Property-based tests with Hypothesis
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hive_protocol.inference.kalman import (
    extract_filtered_states,
    fit_kalman_filter,
    get_noise_estimates,
)


class TestFitKalmanFilterValidation:
    """Tests for input validation in fit_kalman_filter."""

    def test_empty_observations_raises_error(self) -> None:
        """Should raise ValueError for empty observations."""
        with pytest.raises(ValueError, match="cannot be empty"):
            fit_kalman_filter(np.array([]))

    def test_nan_observations_raises_error(self) -> None:
        """Should raise ValueError for observations containing NaN."""
        obs = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="cannot contain NaN"):
            fit_kalman_filter(obs)

    def test_accepts_list_input(self) -> None:
        """Should accept list input and convert to array."""
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Just check it doesn't raise - use minimal sampling
        model, _ = fit_kalman_filter(obs, n_samples=10, n_tune=10)
        assert model is not None


class TestFitKalmanFilterOutput:
    """Tests for output structure of fit_kalman_filter."""

    @pytest.fixture
    def fitted_model(self, simple_observations: np.ndarray):
        """Fit a model with minimal sampling for fast tests."""
        return fit_kalman_filter(
            simple_observations,
            n_samples=50,
            n_tune=50,
            random_seed=42,
        )

    def test_returns_tuple(self, fitted_model) -> None:
        """Should return a tuple of (model, trace)."""
        model, trace = fitted_model
        assert model is not None
        assert trace is not None

    def test_trace_has_posterior(self, fitted_model) -> None:
        """Trace should contain posterior samples."""
        _, trace = fitted_model
        assert hasattr(trace, "posterior")

    def test_posterior_contains_states(self, fitted_model) -> None:
        """Posterior should contain state estimates."""
        _, trace = fitted_model
        assert "states" in trace.posterior

    def test_posterior_contains_noise_params(self, fitted_model) -> None:
        """Posterior should contain noise parameters."""
        _, trace = fitted_model
        assert "process_noise" in trace.posterior
        assert "measurement_noise" in trace.posterior

    def test_states_shape_matches_observations(
        self, fitted_model, simple_observations: np.ndarray
    ) -> None:
        """State estimates should have same length as observations."""
        _, trace = fitted_model
        states = trace.posterior["states"]
        # Shape is (chains, draws, timesteps)
        assert states.shape[-1] == len(simple_observations)

    def test_reproducibility_with_seed(self, simple_observations: np.ndarray) -> None:
        """Same random_seed should produce identical results."""
        _, trace1 = fit_kalman_filter(
            simple_observations,
            n_samples=20,
            n_tune=20,
            random_seed=123,
        )
        _, trace2 = fit_kalman_filter(
            simple_observations,
            n_samples=20,
            n_tune=20,
            random_seed=123,
        )
        states1 = trace1.posterior["states"].values
        states2 = trace2.posterior["states"].values
        np.testing.assert_array_almost_equal(states1, states2)


class TestExtractFilteredStates:
    """Tests for extract_filtered_states function."""

    @pytest.fixture
    def trace(self, simple_observations: np.ndarray):
        """Get trace from fitted model."""
        _, trace = fit_kalman_filter(
            simple_observations,
            n_samples=50,
            n_tune=50,
            random_seed=42,
        )
        return trace

    def test_returns_dict(self, trace) -> None:
        """Should return a dictionary."""
        result = extract_filtered_states(trace)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, trace) -> None:
        """Should contain mean, median, lower, upper."""
        result = extract_filtered_states(trace)
        assert "mean" in result
        assert "median" in result
        assert "lower" in result
        assert "upper" in result

    def test_arrays_are_correct_length(
        self, trace, simple_observations: np.ndarray
    ) -> None:
        """All arrays should match observation length."""
        result = extract_filtered_states(trace)
        n = len(simple_observations)
        assert len(result["mean"]) == n
        assert len(result["median"]) == n
        assert len(result["lower"]) == n
        assert len(result["upper"]) == n

    def test_credible_interval_ordering(self, trace) -> None:
        """Lower bound should be less than upper bound."""
        result = extract_filtered_states(trace)
        assert np.all(result["lower"] <= result["upper"])

    def test_median_within_interval(self, trace) -> None:
        """Median should be within credible interval."""
        result = extract_filtered_states(trace)
        assert np.all(result["median"] >= result["lower"])
        assert np.all(result["median"] <= result["upper"])


class TestGetNoiseEstimates:
    """Tests for get_noise_estimates function."""

    @pytest.fixture
    def trace(self, simple_observations: np.ndarray):
        """Get trace from fitted model."""
        _, trace = fit_kalman_filter(
            simple_observations,
            n_samples=50,
            n_tune=50,
            random_seed=42,
        )
        return trace

    def test_returns_dict(self, trace) -> None:
        """Should return a dictionary."""
        result = get_noise_estimates(trace)
        assert isinstance(result, dict)

    def test_contains_both_noise_params(self, trace) -> None:
        """Should contain estimates for both noise parameters."""
        result = get_noise_estimates(trace)
        assert "process_noise" in result
        assert "measurement_noise" in result

    def test_contains_summary_statistics(self, trace) -> None:
        """Each parameter should have mean, sd, and HDI."""
        result = get_noise_estimates(trace)
        for param in ["process_noise", "measurement_noise"]:
            assert "mean" in result[param]
            assert "sd" in result[param]
            assert "hdi_3%" in result[param]
            assert "hdi_97%" in result[param]

    def test_estimates_are_positive(self, trace) -> None:
        """Noise estimates should be positive (HalfNormal prior)."""
        result = get_noise_estimates(trace)
        assert result["process_noise"]["mean"] > 0
        assert result["measurement_noise"]["mean"] > 0


class TestKalmanFilterHypothesis:
    """Property-based tests using Hypothesis."""

    @given(
        observations=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=30),
            elements=st.floats(min_value=-100, max_value=100),
        )
    )
    @settings(max_examples=5, deadline=60000)  # 60s deadline for MCMC
    def test_filter_output_length_matches_input(self, observations: np.ndarray) -> None:
        """Filtered states should match observation length."""
        # Skip if any NaN (Hypothesis can generate edge cases)
        if np.any(np.isnan(observations)):
            return

        _, trace = fit_kalman_filter(
            observations,
            n_samples=10,
            n_tune=10,
            random_seed=42,
        )
        states = extract_filtered_states(trace)
        assert len(states["mean"]) == len(observations)

    @given(
        n_steps=st.integers(min_value=5, max_value=20),
        process_prior=st.floats(min_value=0.1, max_value=5.0),
        measurement_prior=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=3, deadline=60000)
    def test_different_priors_produce_valid_output(
        self,
        n_steps: int,
        process_prior: float,
        measurement_prior: float,
    ) -> None:
        """Different prior scales should still produce valid output."""
        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, n_steps)

        _, trace = fit_kalman_filter(
            observations,
            process_variance_prior=process_prior,
            measurement_variance_prior=measurement_prior,
            n_samples=10,
            n_tune=10,
            random_seed=42,
        )

        states = extract_filtered_states(trace)
        assert np.all(np.isfinite(states["mean"]))
        assert np.all(np.isfinite(states["lower"]))
        assert np.all(np.isfinite(states["upper"]))
