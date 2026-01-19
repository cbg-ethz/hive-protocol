# Hive-Protocol Tutorial

A step-by-step guide for workshop participants to set up and explore modern Python practices for computational biology.

## Prerequisites

- Git installed and configured
- Terminal/command line familiarity
- Basic Python knowledge
- GitHub account (for forking)

## Part 1: Environment Setup (10 minutes)

### Step 1: Fork the Repository

1. Go to https://github.com/cbg-ethz/hive-protocol
2. Click **Fork** (top right)
3. Select your account as the destination

### Step 2: Clone Your Fork

```bash
# Clone your fork (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/hive-protocol.git
cd hive-protocol
```

### Step 3: Install Pixi

Pixi is the modern package manager for scientific Python. It handles both conda and PyPI packages.

```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Restart your terminal or run:
source ~/.bashrc  # or ~/.zshrc
```

Verify installation:
```bash
pixi --version
```

### Step 4: Create Environment

```bash
# This creates the environment and installs all dependencies
pixi install

# Verify everything works
pixi run test
```

Expected output: All tests pass (green).

## Part 2: Explore the Package (15 minutes)

### Step 5: Try the Code

Open a Python shell:
```bash
pixi shell
python
```

Run this example:
```python
from hive_protocol.data import simulate_noisy_trajectory
from hive_protocol.inference import fit_kalman_filter, extract_filtered_states

# Generate synthetic data
true_states, observations = simulate_noisy_trajectory(
    n_steps=50,
    process_noise=0.2,
    measurement_noise=0.5,
    seed=42,
)

print(f"Generated {len(observations)} observations")
print(f"True state range: [{true_states.min():.2f}, {true_states.max():.2f}]")

# Fit Kalman filter (this takes ~30 seconds)
model, trace = fit_kalman_filter(
    observations,
    n_samples=500,
    n_tune=250,
    random_seed=42,
)

# Extract results
filtered = extract_filtered_states(trace)
print(f"Filtered state RMSE: {((filtered['mean'] - true_states)**2).mean()**0.5:.4f}")
```

### Step 6: Run the Notebooks

Render and view the tutorial notebooks:
```bash
pixi run docs
open docs/tutorials/index.html  # or xdg-open on Linux
```

Navigate through:
1. **Introduction** - Bayesian basics
2. **Kalman Filter** - State-space models
3. **Diagnostics** - Checking your results

## Part 3: Code Quality Tools (15 minutes)

### Step 7: Install Pre-commit Hooks

Pre-commit runs checks automatically before each commit:
```bash
pixi run hooks
```

### Step 8: See Ruff in Action

Create a file with style issues:
```bash
cat > test_style.py << 'EOF'
import numpy as np
import os
import sys
def bad_function(x,y):
    result=x+y
    return result
EOF
```

Run the linter:
```bash
pixi run lint
```

Auto-fix issues:
```bash
pixi run lint-fix
pixi run format
cat test_style.py  # See the fixed code
rm test_style.py   # Clean up
```

### Step 9: Type Checking

See type errors:
```bash
pixi run typecheck
```

## Part 4: Make a Change (20 minutes)

### Step 10: Create a Feature Branch

```bash
git checkout -b feature/add-linear-trend
```

### Step 11: Add a New Function

Edit `src/hive_protocol/data/simulate.py` and add this function:

```python
def simulate_linear_trend(
    n_steps: int = 100,
    slope: float = 0.1,
    intercept: float = 0.0,
    measurement_noise: float = 0.5,
    seed: int | None = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a linear trend with noisy observations.

    Args:
        n_steps: Number of time steps.
        slope: Slope of the linear trend.
        intercept: Y-intercept.
        measurement_noise: Standard deviation of observation noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (true_values, observations).
    """
    rng = np.random.default_rng(seed)

    # True linear trend
    time = np.arange(n_steps, dtype=np.float64)
    true_values = intercept + slope * time

    # Add noise
    observations = true_values + rng.normal(0, measurement_noise, n_steps)

    return true_values, observations
```

### Step 12: Update Exports

Edit `src/hive_protocol/data/__init__.py`:
```python
from hive_protocol.data.simulate import (
    simulate_noisy_trajectory,
    simulate_linear_trend,  # Add this line
)

__all__ = [
    "simulate_noisy_trajectory",
    "simulate_linear_trend",  # Add this line
]
```

### Step 13: Write a Test

Create `tests/test_linear_trend.py`:

```python
"""Tests for linear trend simulation."""

import numpy as np
import pytest

from hive_protocol.data import simulate_linear_trend


def test_simulate_linear_trend_shape():
    """Output should have correct shape."""
    true_vals, obs = simulate_linear_trend(n_steps=50)
    assert true_vals.shape == (50,)
    assert obs.shape == (50,)


def test_simulate_linear_trend_slope():
    """Slope should be approximately correct."""
    true_vals, _ = simulate_linear_trend(
        n_steps=100,
        slope=0.5,
        intercept=0.0,
        measurement_noise=0.0,
    )
    # First value should be 0, last should be ~49.5
    assert np.isclose(true_vals[0], 0.0)
    assert np.isclose(true_vals[-1], 49.5)


def test_simulate_linear_trend_reproducibility():
    """Same seed should give same results."""
    result1 = simulate_linear_trend(seed=123)
    result2 = simulate_linear_trend(seed=123)
    np.testing.assert_array_equal(result1[0], result2[0])
    np.testing.assert_array_equal(result1[1], result2[1])
```

### Step 14: Run All Checks

```bash
# Run your new test
pixi run test tests/test_linear_trend.py

# Run all checks
pixi run check
```

### Step 15: Commit Your Change

```bash
git add .
git commit -m "feat: add linear trend simulation function

- Added simulate_linear_trend to data module
- Includes deterministic trend with Gaussian noise
- Full test coverage"
```

The pre-commit hooks will run automatically and may auto-fix formatting.

## Part 5: Explore Snakemake (Optional, 10 minutes)

### Step 16: See the Workflow

```bash
# Dry run - see what would execute
pixi run workflow-dry
```

### Step 17: Run a Small Experiment

Edit `workflow/config/params.yaml` to reduce the parameter grid:
```yaml
parameter_grid:
  process_noise:
    - 0.1
    - 0.5
  measurement_noise:
    - 0.2
    - 0.8
```

Run the workflow:
```bash
pixi run workflow
```

Check results:
```bash
ls results/
```

## Summary

You've learned to:

1. **Set up** a modern Python environment with pixi
2. **Use** Bayesian inference with PyMC
3. **Apply** code quality tools (Ruff, mypy, pre-commit)
4. **Follow** the feature branch workflow
5. **Write** tests with pytest
6. **Run** reproducible workflows with Snakemake

## Next Steps

- Read through the Quarto notebooks in detail
- Fork and customize for your own projects
- Share with your lab mates!

## Troubleshooting

### Pixi install fails

```bash
# Clear cache and retry
rm -rf .pixi
pixi install
```

### Tests fail with import errors

```bash
# Ensure package is installed in editable mode
pixi install
```

### Pre-commit hooks fail

```bash
# Update hooks
pixi run pre-commit autoupdate
pixi run hooks-all
```

### PyMC sampling is slow

This is normal for MCMC. For workshop purposes, reduce `n_samples`:
```python
model, trace = fit_kalman_filter(obs, n_samples=100, n_tune=50)
```
