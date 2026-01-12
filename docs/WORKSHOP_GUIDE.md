# Workshop Guide: Modern Python for Computational Biology

**D-BSSE Wise Symposium Workshop**
**January 21-23, 2026**
**Format: 20 min presentation + 20 min hands-on**

---

## Pre-Workshop Preparation

### 1 Week Before

- [ ] Send participants the [TUTORIAL.md](TUTORIAL.md) with setup instructions
- [ ] Ask them to complete Steps 1-4 (environment setup) beforehand
- [ ] Prepare backup USB drives with offline installers
- [ ] Test the repository on fresh machines (Mac, Linux)
- [ ] Prepare slides from outline below

### Day Before

- [ ] Verify projector and screen sharing work
- [ ] Have the repository open in VS Code
- [ ] Pre-run all notebook cells (for cached outputs)
- [ ] Prepare quick reference cards for common commands

---

## Workshop Schedule

### Day 1 (January 21): Foundations

| Time | Topic | Materials |
|------|-------|-----------|
| 0:00-0:20 | **Presentation**: Why Modern Python | Slides |
| 0:20-0:40 | **Hands-on**: Environment Setup | TUTORIAL Steps 1-4 |

### Day 2 (January 22): Bayesian Inference

| Time | Topic | Materials |
|------|-------|-----------|
| 0:00-0:20 | **Presentation**: Bayesian Concepts | Slides + `01_introduction.qmd` |
| 0:20-0:40 | **Hands-on**: Kalman Filter | `02_kalman_filter.qmd` |

### Day 3 (January 23): Workflows & Quality

| Time | Topic | Materials |
|------|-------|-----------|
| 0:00-0:20 | **Presentation**: Git + CI/CD | Slides |
| 0:20-0:40 | **Hands-on**: Make a Change | TUTORIAL Part 4 |

---

## Presentation Outline

### Day 1: Why Modern Python (20 min)

**Slide 1: The Problem**
- Show a real "bad" repository (anonymized)
- No tests, no type hints, requirements.txt with conflicts
- "Works on my machine" syndrome

**Slide 2: The 2025 Landscape**
- Tool consolidation: Ruff won, pixi emerged
- Performance gains: 10-100x faster package installs
- Show the Old Way vs Modern Way table from README

**Slide 3: Why Pixi?**
- Bioinformatics needs conda (samtools, bedtools)
- Pixi handles conda + PyPI seamlessly
- Demo: `pixi install` vs traditional setup

**Slide 4: Why Ruff?**
- One tool replaces many
- Speed demo: lint 1000 files in < 1 second
- Live: show auto-fix in action

**Slide 5: The Repository Structure**
- Walk through the directory tree
- Explain src/ layout
- Show pyproject.toml as the single source of truth

### Day 2: Bayesian Concepts (20 min)

**Slide 1: What is Bayesian Inference?**
- Prior + Likelihood → Posterior
- Visual: show updating belief with data

**Slide 2: The Kalman Filter Problem**
- Hidden states, noisy observations
- Applications: GPS, gene expression, cell tracking

**Slide 3: PyMC Syntax**
- Show the model code
- Explain context managers
- Point out the probabilistic programming paradigm

**Slide 4: Diagnostics Matter**
- R-hat, ESS, divergences
- "If you don't check diagnostics, you don't have results"

**Slide 5: Polars for Results**
- Why not pandas? (Performance)
- Lazy evaluation example
- Tidy data principles

### Day 3: Git + CI/CD (20 min)

**Slide 1: Git for Researchers**
- Feature branches protect main
- Commits tell a story
- Show good vs bad commit messages

**Slide 2: Pre-commit Hooks**
- Automated quality gate
- Demo: commit with style errors → auto-fixed

**Slide 3: GitHub Actions**
- CI runs tests on every push
- Show the workflow file
- Badge on README = instant trust

**Slide 4: The Development Loop**
- Branch → Code → Test → Commit → PR → Merge
- Live demo of the full cycle

**Slide 5: Fork and Customize**
- This template is meant to be forked
- Steps to adapt for your project
- Call to action: apply to your current code

---

## Live Coding Script

### Setup Demo (5 min)

```bash
# Show pixi speed
time pixi install  # Should be < 30 seconds

# Show what was installed
pixi list

# Activate and verify
pixi shell
python -c "import pymc; print(pymc.__version__)"
```

### Ruff Demo (3 min)

```bash
# Create messy code
cat > demo.py << 'EOF'
import numpy as np
import os
import sys
def f(x,y):
    z=x+y
    return z
EOF

# Show problems
pixi run ruff check demo.py

# Auto-fix
pixi run ruff check demo.py --fix
pixi run ruff format demo.py

# Show result
cat demo.py
rm demo.py
```

### Kalman Filter Demo (5 min)

```python
# In Python shell
from hive_protocol.data import simulate_noisy_trajectory
from hive_protocol.inference import (
    fit_kalman_filter,
    extract_filtered_states,
    generate_diagnostic_report,
)

# Simulate
states, obs = simulate_noisy_trajectory(n_steps=50, seed=42)
print(f"Range: {states.min():.2f} to {states.max():.2f}")

# Fit (reduce samples for speed)
model, trace = fit_kalman_filter(obs, n_samples=200, n_tune=100)

# Get results
filtered = extract_filtered_states(trace)

# Show diagnostics
print(generate_diagnostic_report(trace, states, filtered))
```

### Git Workflow Demo (5 min)

```bash
# Create branch
git checkout -b demo/add-feature

# Make a change (use pre-prepared code)
# ... edit a file ...

# Stage and commit
git add .
git commit -m "demo: show the workflow"

# Watch pre-commit run!
# If it auto-fixes, recommit

# Push (if time)
git push -u origin demo/add-feature
```

---

## Common Troubleshooting

### "pixi: command not found"

```bash
# Add to PATH
export PATH="$HOME/.pixi/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

### "ModuleNotFoundError: hive_protocol"

```bash
# Ensure editable install
pixi install
# Or manually
pip install -e .
```

### Slow PyMC sampling

Reduce samples for demos:
```python
fit_kalman_filter(obs, n_samples=100, n_tune=50)
```

### Pre-commit fails repeatedly

```bash
# Skip for now
git commit --no-verify -m "message"
# But explain why this is bad practice!
```

### Git merge conflicts

For the workshop, suggest:
```bash
git stash
git pull
git stash pop
```

---

## Exercise Solutions

### Exercise 1: High Measurement Noise

```python
states, obs = simulate_noisy_trajectory(
    n_steps=100,
    process_noise=0.1,
    measurement_noise=2.0,
    seed=42,
)
model, trace = fit_kalman_filter(obs, n_samples=500)
filtered = extract_filtered_states(trace)

# The credible intervals should be wider
# Coverage should still be ~94%
```

### Exercise 2: Fewer Observations

```python
states, obs = simulate_noisy_trajectory(n_steps=20, seed=42)
model, trace = fit_kalman_filter(obs)

# ESS will be lower
# R-hat might be slightly higher
# More uncertainty in estimates
```

### Exercise 3: Wrong Priors

```python
# True process_noise = 0.1, but prior centered at 10
model, trace = fit_kalman_filter(
    obs,
    process_variance_prior=10.0,
)
# Data should overwhelm the prior with enough observations
# Check if estimates still reasonable
```

---

## Post-Workshop

### Follow-up Email

Send within 24 hours:
- Link to the repository
- Link to rendered documentation
- [LEARNING_PATH.md](LEARNING_PATH.md) for continued learning
- Feedback form

### Metrics to Track

- How many participants forked the repo?
- How many opened issues/PRs?
- Pre/post survey on tool familiarity

---

## Backup Plans

### If internet fails

- Have repository cloned on USB drives
- Pre-download all dependencies as a pixi tarball:
  ```bash
  pixi pack
  ```

### If time runs short

Priority order:
1. Environment setup (must complete)
2. Run tests (demonstrates it works)
3. View one notebook (shows the content)
4. Skip live coding, show pre-recorded video

### If participants are stuck

- Pair programming: faster participants help slower ones
- Have TAs circulate
- Provide pre-configured cloud environment (GitHub Codespaces)
