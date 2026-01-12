"""Tests for inference diagnostics module.

Tests cover:
- Convergence checking
- Prediction error computation
- Performance summarization
- Report generation
"""

import numpy as np
import polars as pl
import pytest

from hive_protocol.data import simulate_noisy_trajectory
from hive_protocol.inference import fit_kalman_filter
from hive_protocol.inference.diagnostics import (
    check_convergence,
    compute_prediction_errors,
    generate_diagnostic_report,
    summarize_filter_performance,
)
from hive_protocol.inference.kalman import extract_filtered_states


class TestCheckConvergence:
    """Tests for check_convergence function."""

    @pytest.fixture
    def trace(self) -> None:
        """Get trace from a simple model fit."""
        _, observations = simulate_noisy_trajectory(n_steps=30, seed=42)
        _, trace = fit_kalman_filter(
            observations,
            n_samples=100,
            n_tune=100,
            random_seed=42,
        )
        return trace

    def test_returns_dict(self, trace) -> None:
        """Should return a dictionary."""
        result = check_convergence(trace)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, trace) -> None:
        """Should contain all expected diagnostic keys."""
        result = check_convergence(trace)
        assert "converged" in result
        assert "rhat_max" in result
        assert "ess_min" in result
        assert "n_divergences" in result
        assert "warnings" in result

    def test_converged_is_boolean(self, trace) -> None:
        """converged should be a boolean value."""
        result = check_convergence(trace)
        assert isinstance(result["converged"], bool)

    def test_warnings_is_list(self, trace) -> None:
        """warnings should be a list."""
        result = check_convergence(trace)
        assert isinstance(result["warnings"], list)

    def test_rhat_is_reasonable(self, trace) -> None:
        """R-hat should be close to 1 for converged chains."""
        result = check_convergence(trace)
        # Well-behaved models should have R-hat < 1.1
        assert result["rhat_max"] < 1.5

    def test_ess_is_positive(self, trace) -> None:
        """ESS should be positive."""
        result = check_convergence(trace)
        assert result["ess_min"] > 0

    def test_custom_thresholds(self, trace) -> None:
        """Should respect custom threshold parameters."""
        # Very strict thresholds should trigger warnings
        result = check_convergence(
            trace,
            rhat_threshold=1.0,  # Impossible to achieve
            ess_threshold=10000,  # Very high
        )
        # Should have warnings due to strict thresholds
        assert len(result["warnings"]) > 0


class TestComputePredictionErrors:
    """Tests for compute_prediction_errors function."""

    @pytest.fixture
    def test_data(self):
        """Generate test data with known true states."""
        true_states, observations = simulate_noisy_trajectory(n_steps=30, seed=42)
        _, trace = fit_kalman_filter(
            observations,
            n_samples=50,
            n_tune=50,
            random_seed=42,
        )
        filtered = extract_filtered_states(trace)
        return true_states, filtered

    def test_returns_polars_dataframe(self, test_data) -> None:
        """Should return a Polars DataFrame."""
        true_states, filtered = test_data
        result = compute_prediction_errors(true_states, filtered)
        assert isinstance(result, pl.DataFrame)

    def test_contains_expected_columns(self, test_data) -> None:
        """Should contain all expected columns."""
        true_states, filtered = test_data
        result = compute_prediction_errors(true_states, filtered)
        expected_cols = {
            "timestep",
            "true_state",
            "estimate",
            "error",
            "abs_error",
            "in_credible_interval",
        }
        assert set(result.columns) == expected_cols

    def test_correct_number_of_rows(self, test_data) -> None:
        """Should have one row per timestep."""
        true_states, filtered = test_data
        result = compute_prediction_errors(true_states, filtered)
        assert len(result) == len(true_states)

    def test_error_calculation(self, test_data) -> None:
        """Error should be estimate - true."""
        true_states, filtered = test_data
        result = compute_prediction_errors(true_states, filtered)
        expected_error = filtered["mean"] - true_states
        np.testing.assert_array_almost_equal(result["error"].to_numpy(), expected_error)

    def test_abs_error_is_positive(self, test_data) -> None:
        """Absolute error should be non-negative."""
        true_states, filtered = test_data
        result = compute_prediction_errors(true_states, filtered)
        assert (result["abs_error"] >= 0).all()

    def test_credible_interval_coverage(self, test_data) -> None:
        """Some true states should be in credible interval."""
        true_states, filtered = test_data
        result = compute_prediction_errors(true_states, filtered)
        # With a 94% CI, we expect high coverage
        coverage = result["in_credible_interval"].mean()
        # At least 50% should be covered (very loose check)
        assert coverage > 0.5


class TestSummarizeFilterPerformance:
    """Tests for summarize_filter_performance function."""

    @pytest.fixture
    def errors_df(self):
        """Generate errors DataFrame for testing."""
        true_states, observations = simulate_noisy_trajectory(n_steps=30, seed=42)
        _, trace = fit_kalman_filter(
            observations,
            n_samples=50,
            n_tune=50,
            random_seed=42,
        )
        filtered = extract_filtered_states(trace)
        return compute_prediction_errors(true_states, filtered)

    def test_returns_dict(self, errors_df) -> None:
        """Should return a dictionary."""
        result = summarize_filter_performance(errors_df)
        assert isinstance(result, dict)

    def test_contains_expected_metrics(self, errors_df) -> None:
        """Should contain all expected metric keys."""
        result = summarize_filter_performance(errors_df)
        assert "rmse" in result
        assert "mae" in result
        assert "bias" in result
        assert "coverage" in result
        assert "max_error" in result

    def test_rmse_is_positive(self, errors_df) -> None:
        """RMSE should be non-negative."""
        result = summarize_filter_performance(errors_df)
        assert result["rmse"] >= 0

    def test_mae_is_positive(self, errors_df) -> None:
        """MAE should be non-negative."""
        result = summarize_filter_performance(errors_df)
        assert result["mae"] >= 0

    def test_coverage_is_fraction(self, errors_df) -> None:
        """Coverage should be between 0 and 1."""
        result = summarize_filter_performance(errors_df)
        assert 0 <= result["coverage"] <= 1

    def test_rmse_geq_mae(self, errors_df) -> None:
        """RMSE should be >= MAE (mathematical property)."""
        result = summarize_filter_performance(errors_df)
        assert result["rmse"] >= result["mae"] - 1e-10  # Allow small tolerance


class TestGenerateDiagnosticReport:
    """Tests for generate_diagnostic_report function."""

    @pytest.fixture
    def test_data(self):
        """Generate test data for report."""
        true_states, observations = simulate_noisy_trajectory(n_steps=30, seed=42)
        _, trace = fit_kalman_filter(
            observations,
            n_samples=50,
            n_tune=50,
            random_seed=42,
        )
        filtered = extract_filtered_states(trace)
        return trace, true_states, filtered

    def test_returns_string(self, test_data) -> None:
        """Should return a string."""
        trace, true_states, filtered = test_data
        result = generate_diagnostic_report(trace, true_states, filtered)
        assert isinstance(result, str)

    def test_contains_convergence_section(self, test_data) -> None:
        """Report should include convergence diagnostics."""
        trace, true_states, filtered = test_data
        result = generate_diagnostic_report(trace, true_states, filtered)
        assert "CONVERGENCE" in result
        assert "R-hat" in result
        assert "ESS" in result

    def test_contains_performance_section(self, test_data) -> None:
        """Report should include performance metrics when provided."""
        trace, true_states, filtered = test_data
        result = generate_diagnostic_report(trace, true_states, filtered)
        assert "PERFORMANCE" in result
        assert "RMSE" in result
        assert "Coverage" in result

    def test_works_without_ground_truth(self, test_data) -> None:
        """Should work when true states not provided."""
        trace, _, _ = test_data
        result = generate_diagnostic_report(trace)
        assert isinstance(result, str)
        assert "CONVERGENCE" in result
        # Should not have performance section
        assert "PERFORMANCE" not in result
