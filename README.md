```
═══════════════════════════════════════════════════════════════════════════════

  ⬢ HIVE-PROTOCOL ⬢  RESEARCH SOFTWARE CONTAINMENT STANDARD

═══════════════════════════════════════════════════════════════════════════════
```

<div align="center">

```
        ⬢ ⬢ ⬢ ⬢ ⬢ ⬢ ⬢
        ⬢ ⬢ ⬢ ⬢ ⬢ ⬢ ⬢
        ⬢ ⬢ ⬢ ⬢ ⬢ ⬢ ⬢
```

**THE DEFINITIVE PROTOCOL FOR PYTHON IN COMPUTATIONAL BIOLOGY**

`PROTOCOL-001` | `v1.0.0`

[![CI](https://github.com/cbg-ethz/hive-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/cbg-ethz/hive-protocol/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pixi](https://img.shields.io/badge/pixi-managed-yellow.svg)](https://pixi.sh/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

```
⚠ WARNING: Unstructured research code detected
→ SOLUTION: Deploy HIVE-PROTOCOL
```

---

## PROTOCOL OBJECTIVE

Transform research scripts into production-grade software.

```
PROTOCOL COMPONENTS:
[✓] Dependency Management    [✓] Code Quality
[✓] Testing Framework        [✓] CI/CD Pipeline
[✓] Type Safety              [✓] Documentation
[✓] Configuration            [✓] Reproducibility
```

---

## DEPLOYMENT SEQUENCE

```bash
# 1. CLONE
git clone https://github.com/cbg-ethz/hive-protocol.git
cd hive-protocol

# 2. INSTALL PIXI (if not present)
curl -fsSL https://pixi.sh/install.sh | bash

# 3. INITIALIZE ENVIRONMENT
pixi install

# 4. VERIFY DEPLOYMENT
pixi run test
```

**STATUS:** Environment operational.

---

## TECHNOLOGY MATRIX

| Category | Old Protocol | Modern Protocol | Improvement |
|----------|--------------|-----------------|-------------|
| **Package Management** | pip + requirements.txt | **pixi** | 10-100x faster |
| **Code Quality** | Black + flake8 + isort | **Ruff** | 30-100x faster |
| **Data Processing** | pandas | **Polars** | 5-50x faster |
| **Notebooks** | Jupyter | **Quarto** | Git-friendly |
| **Testing** | unittest | **pytest + Hypothesis** | Property-based |
| **Configuration** | argparse / dict | **Pydantic** | Auto-validation |
| **Type Checking** | None | **mypy + Pyright** | Static analysis |

---

## DIRECTORY STRUCTURE

```
hive-protocol/
├── src/hive_protocol/       # SOURCE CODE (src layout)
│   ├── inference/           # Kalman filter + diagnostics
│   └── data/                # Data simulation
├── tests/                   # PYTEST + HYPOTHESIS
├── notebooks/               # QUARTO TUTORIALS
│   ├── 01_introduction.qmd
│   ├── 02_kalman_filter.qmd
│   └── 03_diagnostics.qmd
├── workflow/                # SNAKEMAKE PIPELINE
│   ├── Snakefile
│   └── config/params.yaml
├── docs/                    # WORKSHOP MATERIALS
├── pyproject.toml           # SINGLE SOURCE OF TRUTH
├── pixi.toml                # ENVIRONMENT SPEC
└── .pre-commit-config.yaml  # QUALITY HOOKS
```

---

## OPERATIONS MANUAL

```bash
# TESTING
pixi run test              # Run test suite
pixi run test-cov          # Run with coverage report

# CODE QUALITY
pixi run lint              # Check code style
pixi run lint-fix          # Auto-fix issues
pixi run format            # Format code

# TYPE CHECKING
pixi run typecheck         # Pyright (fast, local)
pixi run typecheck-ci      # mypy (stable, CI)

# DOCUMENTATION
pixi run docs              # Render notebooks
pixi run slides            # Render presentation

# WORKFLOW
pixi run workflow          # Execute Snakemake pipeline
pixi run workflow-dry      # Dry run (preview)

# MAINTENANCE
pixi run check             # Run ALL quality checks
pixi run hooks             # Install pre-commit hooks
```

---

## PROTOCOL ADAPTATION

### STEPS

1. **FORK** this repository
2. **RENAME** `hive_protocol` → `your_project`
3. **UPDATE** `pyproject.toml` metadata
4. **REPLACE** Kalman filter with your domain logic
5. **RETAIN** testing + CI patterns

### PRESERVE

```
[✓] src/ layout              [✓] pyproject.toml structure
[✓] Test organization        [✓] CI/CD workflows
[✓] Pre-commit config        [✓] Quarto notebooks pattern
```

---

## WORKSHOP MATERIALS

**CBG Retreat 2026** | ETH Zurich | January 21-23, 2026

| Resource | Location |
|----------|----------|
| Slides | `docs/slides.qmd` |
| Tutorial | `docs/TUTORIAL.md` |
| Render | `pixi run slides` |

---

## EXTERNAL REFERENCES

| Tool | Documentation |
|------|---------------|
| Pixi | [pixi.sh](https://pixi.sh) |
| Ruff | [docs.astral.sh/ruff](https://docs.astral.sh/ruff) |
| Pydantic | [docs.pydantic.dev](https://docs.pydantic.dev) |
| Hypothesis | [hypothesis.readthedocs.io](https://hypothesis.readthedocs.io) |
| Quarto | [quarto.org](https://quarto.org) |
| PyMC | [pymc.io](https://www.pymc.io) |

---

## CONTRIBUTING

```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/enhancement

# 3. Make changes
# 4. Verify compliance
pixi run check

# 5. Commit (pre-commit enforces standards)
git commit -m "feat: add enhancement"

# 6. Push and create PR
git push -u origin feature/enhancement
```

---

## LICENSE

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

```
═══════════════════════════════════════════════════════════════════════════════
  HIVE-PROTOCOL | RESEARCH SOFTWARE CONTAINMENT STANDARD | STATUS: OPERATIONAL
═══════════════════════════════════════════════════════════════════════════════
```

*Built with modern Python for computational biology*

**CBG-ETH Zurich**

</div>
