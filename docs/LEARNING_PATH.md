# Learning Path: Modern Python for Computational Biology

A 4-week progression from workshop attendee to confident practitioner.

---

## Overview

| Week | Focus | Key Skills |
|------|-------|------------|
| 1 | Git + Code Quality | Branches, Ruff, basic tests |
| 2 | Testing + Types | pytest, Hypothesis, type hints |
| 3 | Data + Notebooks | Polars, Quarto, reproducibility |
| 4 | Bayesian + Workflows | PyMC, Snakemake, CI/CD |

---

## Week 1: Git and Code Quality

**Goal**: Establish professional version control and code formatting habits.

### Learning Objectives

- [ ] Understand feature branch workflow
- [ ] Use Ruff for automatic formatting
- [ ] Write meaningful commit messages
- [ ] Set up pre-commit hooks

### Daily Practice (15-30 min/day)

**Day 1: Git Basics Review**
```bash
# Create a practice repository
mkdir git-practice && cd git-practice
git init

# Create some files, make commits
echo "# Practice" > README.md
git add README.md
git commit -m "Initial commit"
```

**Day 2: Feature Branches**
```bash
# Create a feature branch
git checkout -b feature/add-something

# Make changes, commit
git add .
git commit -m "feat: add something useful"

# Merge back to main
git checkout main
git merge feature/add-something
```

**Day 3: Ruff Setup**
```bash
# In any Python project
pip install ruff

# Check code
ruff check .

# Format code
ruff format .

# Add to pyproject.toml
```

**Day 4: Pre-commit Hooks**
```bash
# Install
pip install pre-commit

# Create .pre-commit-config.yaml (copy from hive-protocol)

# Install hooks
pre-commit install

# Test with a commit
```

**Day 5: Practice**
- Fork hive-protocol
- Create a branch
- Make a small change
- Commit and see hooks run

### Resources

- [Pro Git Book](https://git-scm.com/book/en/v2) - Chapters 1-3
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Week 2: Testing and Type Hints

**Goal**: Write tests for your code and use type hints for documentation.

### Learning Objectives

- [ ] Write unit tests with pytest
- [ ] Use fixtures for test data
- [ ] Try property-based testing with Hypothesis
- [ ] Add type hints to function signatures

### Daily Practice

**Day 1: First pytest Tests**
```python
# test_example.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
```

Run with: `pytest test_example.py -v`

**Day 2: Fixtures**
```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# test_stats.py
def test_mean(sample_data):
    assert np.mean(sample_data) == 3.0
```

**Day 3: Type Hints**
```python
from numpy.typing import NDArray
import numpy as np

def calculate_mean(data: NDArray[np.float64]) -> float:
    """Calculate the arithmetic mean."""
    return float(np.mean(data))

def process_data(
    values: list[float],
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Process a list of values."""
    arr = np.array(values)
    if normalize:
        arr = arr / arr.max()
    return arr
```

**Day 4: Hypothesis**
```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.floats(allow_nan=False), min_size=1))
def test_mean_bounds(values):
    """Mean should be within data bounds."""
    arr = np.array(values)
    mean = np.mean(arr)
    assert arr.min() <= mean <= arr.max()
```

**Day 5: Apply to Your Code**
- Pick a function from your research code
- Add type hints
- Write 3 tests for it

### Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Python Type Hints Guide](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

---

## Week 3: Data and Notebooks

**Goal**: Process data efficiently and create reproducible reports.

### Learning Objectives

- [ ] Use Polars for DataFrame operations
- [ ] Write Quarto notebooks
- [ ] Understand lazy evaluation
- [ ] Create reproducible analysis documents

### Daily Practice

**Day 1: Polars Basics**
```python
import polars as pl

# Create DataFrame
df = pl.DataFrame({
    "gene": ["BRCA1", "TP53", "EGFR"],
    "expression": [5.2, 3.1, 7.8],
    "condition": ["treated", "control", "treated"],
})

# Basic operations
print(df.filter(pl.col("expression") > 4))
print(df.group_by("condition").agg(pl.col("expression").mean()))
```

**Day 2: Lazy Evaluation**
```python
# Lazy frames don't execute until .collect()
lazy_df = pl.scan_csv("large_file.csv")

result = (
    lazy_df
    .filter(pl.col("value") > 0)
    .group_by("category")
    .agg(pl.col("value").sum())
    .collect()  # Execution happens here
)
```

**Day 3: Quarto Basics**
Create `analysis.qmd`:
```markdown
---
title: "My Analysis"
format: html
jupyter: python3
---

## Introduction

This is my analysis.

```{python}
import polars as pl
df = pl.DataFrame({"x": [1, 2, 3]})
print(df)
```
```

Render: `quarto render analysis.qmd`

**Day 4: Quarto Features**
```yaml
---
execute:
  freeze: auto  # Cache results
  echo: true    # Show code
---
```

**Day 5: Your Data**
- Load a dataset from your research
- Process with Polars
- Create a Quarto report

### Resources

- [Polars User Guide](https://pola.rs/)
- [Quarto Documentation](https://quarto.org/docs/guide/)
- [From Pandas to Polars](https://pola-rs.github.io/polars/py-polars/html/reference/index.html)

---

## Week 4: Bayesian Inference and Workflows

**Goal**: Apply Bayesian methods and automate pipelines.

### Learning Objectives

- [ ] Build simple PyMC models
- [ ] Interpret posterior distributions
- [ ] Create Snakemake workflows
- [ ] Set up GitHub Actions CI

### Daily Practice

**Day 1: PyMC Basics**
```python
import pymc as pm
import numpy as np

# Data
data = np.random.normal(5, 2, size=100)

# Model
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(1000)

# Results
import arviz as az
print(az.summary(trace))
```

**Day 2: Diagnostics**
```python
# Always check these!
print(az.summary(trace, var_names=["mu", "sigma"]))

# Look for:
# - R-hat close to 1.0
# - ESS > 400
# - No divergences
```

**Day 3: Snakemake Basics**
```python
# Snakefile
rule all:
    input: "results/analysis.html"

rule process:
    input: "data/raw.csv"
    output: "data/processed.csv"
    shell: "python scripts/process.py {input} {output}"

rule analyze:
    input: "data/processed.csv"
    output: "results/analysis.html"
    shell: "quarto render analysis.qmd"
```

Run: `snakemake --cores 1`

**Day 4: GitHub Actions**
Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
      - run: pixi run test
```

**Day 5: Full Pipeline**
- Create a mini-project combining everything
- Data → Processing → Analysis → Report
- Automate with Snakemake
- Test with CI

### Resources

- [PyMC Documentation](https://www.pymc.io/welcome.html)
- [Statistical Rethinking (PyMC)](https://github.com/pymc-devs/pymc-resources)
- [Snakemake Tutorial](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## Ongoing Practice

### Monthly Goals

1. **Refactor one old project** using these practices
2. **Review a colleague's PR** using the checklist
3. **Write a blog post or lab presentation** about what you learned

### Checklist for New Projects

- [ ] Use src/ layout
- [ ] Create pyproject.toml
- [ ] Set up pixi environment
- [ ] Install pre-commit hooks
- [ ] Write tests from day 1
- [ ] Use type hints
- [ ] Set up GitHub Actions

### Keep Learning

- Join the [PyMC Discourse](https://discourse.pymc.io/)
- Follow [Scientific Python](https://scientific-python.org/)
- Attend PyData conferences
- Contribute to open source

---

## Quick Reference

### Commands You'll Use Daily

```bash
# Environment
pixi install
pixi shell
pixi run test

# Code quality
pixi run lint
pixi run format
pixi run check

# Git
git checkout -b feature/name
git add .
git commit -m "type: message"
git push -u origin feature/name

# Documentation
quarto render notebook.qmd
quarto preview notebook.qmd
```

### Good Commit Message Examples

```
feat: add data validation function
fix: handle empty input arrays
docs: update installation instructions
test: add edge case tests for parser
refactor: simplify data loading logic
```

### Type Hint Cheatsheet

```python
# Basic types
def greet(name: str) -> str: ...
def add(a: int, b: int) -> int: ...
def process(data: list[float]) -> float: ...

# Optional
def fetch(url: str, timeout: int | None = None) -> str: ...

# NumPy
from numpy.typing import NDArray
import numpy as np
def analyze(data: NDArray[np.float64]) -> float: ...

# Polars
import polars as pl
def transform(df: pl.DataFrame) -> pl.DataFrame: ...
```

---

*Keep practicing, and soon these tools will feel natural!*
