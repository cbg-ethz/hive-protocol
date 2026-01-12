"""Posterior diagnostics for Bayesian inference results.

This module provides utilities for assessing the quality of MCMC
sampling and the reliability of posterior estimates. Good diagnostics
are essential for trusting Bayesian inference results.

Key diagnostics include:
- R-hat: Measures chain convergence (should be < 1.01)
- ESS: Effective sample size (should be > 400 for reliable estimates)
- Divergences: NUTS sampler warnings (should be 0)

Example:
    >>> from hive_protocol.inference import fit_kalman_filter
    >>> from hive_protocol.inference.diagnostics import check_convergence
    >>> model, trace = fit_kalman_filter(observations)
    >>> report = check_convergence(trace)
    >>> print(f"All chains converged: {report['converged']}")
"""

import warnings
from typing import Any

import arviz as az
import numpy as np
import polars as pl
from numpy.typing import NDArray


def check_convergence(
    trace: az.InferenceData,
    rhat_threshold: float = 1.01,
    ess_threshold: int = 400,
) -> dict[str, Any]:
    """Check MCMC convergence diagnostics.

    Evaluates whether the posterior samples are reliable by checking
    standard convergence criteria.

    Args:
        trace: ArviZ InferenceData from PyMC sampling.
        rhat_threshold: Maximum acceptable R-hat value.
            Values > 1.01 indicate potential non-convergence.
        ess_threshold: Minimum acceptable effective sample size.
            Values < 400 suggest insufficient sampling.

    Returns:
        Dictionary containing:
        - converged: bool, True if all diagnostics pass
        - rhat_max: Maximum R-hat across all parameters
        - ess_min: Minimum ESS across all parameters
        - n_divergences: Number of divergent transitions
        - warnings: List of diagnostic warning messages

    Example:
        >>> report = check_convergence(trace)
        >>> if not report['converged']:
        ...     for warning in report['warnings']:
        ...         print(f"WARNING: {warning}")
    """
    warnings_list: list[str] = []

    # Get summary statistics
    summary: Any = az.summary(trace)

    # Check R-hat (potential scale reduction factor)
    rhat_values = summary["r_hat"].values
    rhat_max = float(np.nanmax(rhat_values))
    if rhat_max > rhat_threshold:
        warnings_list.append(
            f"R-hat too high: {rhat_max:.3f} > {rhat_threshold} "
            "(chains may not have converged)"
        )

    # Check ESS (effective sample size)
    # Use bulk ESS as it's more relevant for posterior means
    ess_values = summary["ess_bulk"].values
    ess_min = int(np.nanmin(ess_values))
    if ess_min < ess_threshold:
        warnings_list.append(
            f"ESS too low: {ess_min} < {ess_threshold} "
            "(consider running more samples)"
        )

    # Check for divergences (NUTS-specific)
    n_divergences = 0
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        n_divergences = int(trace.sample_stats["diverging"].sum().values)
        if n_divergences > 0:
            warnings_list.append(
                f"Found {n_divergences} divergent transitions "
                "(model may be misspecified or needs reparameterization)"
            )

    converged = len(warnings_list) == 0

    return {
        "converged": converged,
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "n_divergences": n_divergences,
        "warnings": warnings_list,
    }


def compute_prediction_errors(
    true_states: NDArray[np.float64],
    filtered_states: dict[str, NDArray[np.float64]],
) -> pl.DataFrame:
    """Compute prediction errors between true and filtered states.

    Calculates various error metrics useful for assessing filter
    performance when ground truth is available (e.g., simulations).

    Args:
        true_states: Ground truth state trajectory.
        filtered_states: Dictionary from extract_filtered_states()
            containing 'mean', 'median', 'lower', 'upper'.

    Returns:
        Polars DataFrame with columns:
        - timestep: Time index
        - true_state: Ground truth value
        - estimate: Filtered mean estimate
        - error: Signed error (estimate - true)
        - abs_error: Absolute error
        - in_credible_interval: Whether true state is in CI

    Example:
        >>> from hive_protocol.inference.kalman import extract_filtered_states
        >>> states = extract_filtered_states(trace)
        >>> errors = compute_prediction_errors(true_states, states)
        >>> coverage = errors["in_credible_interval"].mean()
        >>> print(f"Credible interval coverage: {coverage:.1%}")
    """
    n_timesteps = len(true_states)
    estimates = filtered_states["mean"]
    lower = filtered_states["lower"]
    upper = filtered_states["upper"]

    # Check if true state falls within credible interval
    in_ci = (true_states >= lower) & (true_states <= upper)

    errors = estimates - true_states
    abs_errors = np.abs(errors)

    return pl.DataFrame(
        {
            "timestep": list(range(n_timesteps)),
            "true_state": true_states.tolist(),
            "estimate": estimates.tolist(),
            "error": errors.tolist(),
            "abs_error": abs_errors.tolist(),
            "in_credible_interval": in_ci.tolist(),
        }
    )


def summarize_filter_performance(
    errors_df: pl.DataFrame,
) -> dict[str, float]:
    """Summarize filter performance metrics.

    Computes aggregate statistics from prediction errors to assess
    overall filter quality.

    Args:
        errors_df: DataFrame from compute_prediction_errors().

    Returns:
        Dictionary containing:
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - bias: Mean signed error (systematic over/under-estimation)
        - coverage: Fraction of true states within credible interval
        - max_error: Maximum absolute error

    Example:
        >>> summary = summarize_filter_performance(errors_df)
        >>> print(f"RMSE: {summary['rmse']:.4f}")
        >>> print(f"Coverage: {summary['coverage']:.1%}")
    """
    errors = errors_df["error"].to_numpy()
    abs_errors = errors_df["abs_error"].to_numpy()

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(abs_errors))
    bias = float(np.mean(errors))
    coverage = float(errors_df["in_credible_interval"].mean())
    max_error = float(np.max(abs_errors))

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "coverage": coverage,
        "max_error": max_error,
    }


def generate_diagnostic_report(
    trace: az.InferenceData,
    true_states: NDArray[np.float64] | None = None,
    filtered_states: dict[str, NDArray[np.float64]] | None = None,
) -> str:
    """Generate a human-readable diagnostic report.

    Produces a text summary suitable for logging or printing that
    covers convergence diagnostics and optionally filter performance.

    Args:
        trace: ArviZ InferenceData from PyMC sampling.
        true_states: Optional ground truth for performance metrics.
        filtered_states: Optional filtered states for performance metrics.

    Returns:
        Formatted string report.

    Example:
        >>> report = generate_diagnostic_report(trace, true_states, filtered)
        >>> print(report)
    """
    lines = [
        "=" * 60,
        "KALMAN FILTER DIAGNOSTIC REPORT",
        "=" * 60,
        "",
    ]

    # Convergence diagnostics
    convergence = check_convergence(trace)
    lines.append("CONVERGENCE DIAGNOSTICS")
    lines.append("-" * 40)
    lines.append(f"  Converged: {'Yes' if convergence['converged'] else 'NO'}")
    lines.append(f"  Max R-hat: {convergence['rhat_max']:.4f}")
    lines.append(f"  Min ESS: {convergence['ess_min']}")
    lines.append(f"  Divergences: {convergence['n_divergences']}")

    if convergence["warnings"]:
        lines.append("")
        lines.append("  Warnings:")
        for warning in convergence["warnings"]:
            lines.append(f"    - {warning}")

    # Performance metrics (if ground truth available)
    if true_states is not None and filtered_states is not None:
        errors_df = compute_prediction_errors(true_states, filtered_states)
        performance = summarize_filter_performance(errors_df)

        lines.append("")
        lines.append("FILTER PERFORMANCE")
        lines.append("-" * 40)
        lines.append(f"  RMSE: {performance['rmse']:.4f}")
        lines.append(f"  MAE: {performance['mae']:.4f}")
        lines.append(f"  Bias: {performance['bias']:.4f}")
        lines.append(f"  CI Coverage: {performance['coverage']:.1%}")
        lines.append(f"  Max Error: {performance['max_error']:.4f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
