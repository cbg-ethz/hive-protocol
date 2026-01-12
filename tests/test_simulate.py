"""Tests for data simulation module.

Tests cover:
- Basic functionality and output shapes
- Parameter validation
- Reproducibility with seeds
- Property-based tests with Hypothesis
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hive_protocol.data import simulate_noisy_trajectory
from hive_protocol.data.simulate import (
    simulate_multiple_trajectories,
    simulate_parameter_sweep,
)


class TestSimulateNoisyTrajectory:
    """Tests for the simulate_noisy_trajectory function."""

    def test_returns_tuple_of_arrays(self) -> None:
        """Should return a tuple of two numpy arrays."""
        result = simulate_noisy_trajectory()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_output_shapes_match(self) -> None:
        """States and observations should have same length."""
        n_steps = 50
        states, observations = simulate_noisy_trajectory(n_steps=n_steps)
        assert states.shape == (n_steps,)
        assert observations.shape == (n_steps,)

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical results."""
        result1 = simulate_noisy_trajectory(seed=42)
        result2 = simulate_noisy_trajectory(seed=42)
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should produce different results."""
        result1 = simulate_noisy_trajectory(seed=42)
        result2 = simulate_noisy_trajectory(seed=43)
        assert not np.allclose(result1[0], result2[0])

    def test_zero_noise_produces_identical_output(self) -> None:
        """With zero noise, states and observations should match."""
        states, observations = simulate_noisy_trajectory(
            process_noise=0.1,
            measurement_noise=0.0,
        )
        np.testing.assert_array_equal(states, observations)

    def test_initial_state_is_respected(self) -> None:
        """Initial state parameter should influence trajectory start."""
        _, obs1 = simulate_noisy_trajectory(initial_state=0.0, seed=42)
        _, obs2 = simulate_noisy_trajectory(initial_state=100.0, seed=42)
        # The trajectories should be offset
        assert abs(obs2.mean() - obs1.mean()) > 50

    def test_negative_n_steps_raises_error(self) -> None:
        """Should raise ValueError for invalid n_steps."""
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            simulate_noisy_trajectory(n_steps=0)

    def test_negative_process_noise_raises_error(self) -> None:
        """Should raise ValueError for negative process_noise."""
        with pytest.raises(ValueError, match="process_noise must be >= 0"):
            simulate_noisy_trajectory(process_noise=-0.1)

    def test_negative_measurement_noise_raises_error(self) -> None:
        """Should raise ValueError for negative measurement_noise."""
        with pytest.raises(ValueError, match="measurement_noise must be >= 0"):
            simulate_noisy_trajectory(measurement_noise=-0.1)


class TestSimulateNoisyTrajectoryHypothesis:
    """Property-based tests using Hypothesis."""

    @given(n_steps=st.integers(min_value=1, max_value=500))
    @settings(max_examples=20)
    def test_output_length_matches_n_steps(self, n_steps: int) -> None:
        """Output length should always match requested n_steps."""
        states, observations = simulate_noisy_trajectory(n_steps=n_steps)
        assert len(states) == n_steps
        assert len(observations) == n_steps

    @given(
        process_noise=st.floats(min_value=0.0, max_value=10.0),
        measurement_noise=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_valid_noise_parameters_dont_raise(
        self, process_noise: float, measurement_noise: float
    ) -> None:
        """Any non-negative noise parameters should work."""
        states, observations = simulate_noisy_trajectory(
            n_steps=10,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        assert not np.any(np.isnan(states))
        assert not np.any(np.isnan(observations))

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=10)
    def test_any_seed_produces_valid_output(self, seed: int) -> None:
        """Any valid seed should produce finite output."""
        states, observations = simulate_noisy_trajectory(n_steps=20, seed=seed)
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(observations))


class TestSimulateMultipleTrajectories:
    """Tests for the simulate_multiple_trajectories function."""

    def test_returns_polars_dataframe(self) -> None:
        """Should return a Polars DataFrame."""
        result = simulate_multiple_trajectories(n_trajectories=3, n_steps=10)
        assert isinstance(result, pl.DataFrame)

    def test_correct_number_of_rows(self) -> None:
        """DataFrame should have n_trajectories * n_steps rows."""
        n_traj = 5
        n_steps = 20
        result = simulate_multiple_trajectories(n_trajectories=n_traj, n_steps=n_steps)
        assert len(result) == n_traj * n_steps

    def test_expected_columns(self) -> None:
        """DataFrame should have all expected columns."""
        result = simulate_multiple_trajectories(n_trajectories=2, n_steps=10)
        expected_cols = {
            "trajectory_id",
            "timestep",
            "true_state",
            "observation",
            "process_noise",
            "measurement_noise",
        }
        assert set(result.columns) == expected_cols

    def test_trajectory_ids_are_correct(self) -> None:
        """Each trajectory should have unique consecutive ID."""
        n_traj = 4
        result = simulate_multiple_trajectories(n_trajectories=n_traj, n_steps=10)
        trajectory_ids = result["trajectory_id"].unique().sort()
        expected = pl.Series(list(range(n_traj)))
        assert trajectory_ids.to_list() == expected.to_list()


class TestSimulateParameterSweep:
    """Tests for the simulate_parameter_sweep function."""

    def test_returns_polars_dataframe(self) -> None:
        """Should return a Polars DataFrame."""
        result = simulate_parameter_sweep(
            process_noise_values=[0.1, 0.5],
            measurement_noise_values=[0.1, 0.5],
            n_steps=10,
        )
        assert isinstance(result, pl.DataFrame)

    def test_all_parameter_combinations(self) -> None:
        """Should have data for all parameter combinations."""
        proc_values = [0.1, 0.5, 1.0]
        meas_values = [0.2, 0.4]
        result = simulate_parameter_sweep(
            process_noise_values=proc_values,
            measurement_noise_values=meas_values,
            n_steps=10,
        )
        # Number of unique experiments should equal product of list lengths
        n_experiments = result["experiment_id"].n_unique()
        assert n_experiments == len(proc_values) * len(meas_values)

    def test_parameter_values_are_recorded(self) -> None:
        """Parameter values should be correctly recorded in DataFrame."""
        proc_values = [0.1, 0.5]
        meas_values = [0.2, 0.8]
        result = simulate_parameter_sweep(
            process_noise_values=proc_values,
            measurement_noise_values=meas_values,
            n_steps=10,
        )
        recorded_proc = set(result["process_noise"].unique().to_list())
        recorded_meas = set(result["measurement_noise"].unique().to_list())
        assert recorded_proc == set(proc_values)
        assert recorded_meas == set(meas_values)
