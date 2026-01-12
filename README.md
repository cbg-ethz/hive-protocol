# Hive-Protocol

**The definitive protocol for Python in computational biology**

A template repository demonstrating modern Python standards for computational biology research. Created for the CBG Retreat 2026 at ETH Zurich (January 21-23, 2026).

---

## Why This Template Exists

The Python ecosystem has matured dramatically, but most research code still uses outdated practices. This template showcases the 2025 state-of-the-art:

| Old Way | Modern Way | Benefit |
|---------|-----------|---------|
| pip + requirements.txt | **pixi** | 10-100x faster, handles conda + PyPI |
| Black + flake8 + isort | **Ruff** | Single tool, 30-100x faster |
| pandas for everything | **Polars** | 5-50x faster for large data |
| Jupyter notebooks | **Quarto** | Clean git diffs, reproducible |
| Manual testing | **pytest + Hypothesis** | Property-based testing |
| No type hints | **Type hints + mypy** | Catch bugs before runtime |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/cbg-ethz/hive-protocol.git
cd hive-protocol

# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Set up environment (installs all dependencies)
pixi install

# Verify installation
pixi run test

# Start exploring
pixi run docs  # Render tutorial notebooks
```

## Technology Stack

### Package Management: [Pixi](https://pixi.sh/)
Handles conda and PyPI packages seamlessly. Essential for bioinformatics where tools like samtools remain conda-only.

### Bayesian Inference: [PyMC 5+](https://www.pymc.io/) *(content example)*
Used here as example scientific content. The Kalman filter demonstrates state-space modeling—replace with your own domain logic.

### Data Processing: [Polars](https://pola.rs/)
DataFrame library built in Rust. 5-50x faster than pandas with lazy evaluation.

### Code Quality: [Ruff](https://docs.astral.sh/ruff/)
Replaces Black, flake8, isort, pyupgrade, and more. Written in Rust, 30-100x faster.

### Documentation: [Quarto](https://quarto.org/)
Reproducible notebooks with clean git diffs. Outputs render on demand.

### Workflow: [Snakemake](https://snakemake.readthedocs.io/)
Reproducible pipeline orchestration with automatic parallelization.

## Repository Structure

```
hive-protocol/
├── src/hive_protocol/       # Source code
│   ├── inference/           # Kalman filter + diagnostics
│   └── data/                # Data simulation
├── tests/                   # pytest + Hypothesis tests
├── notebooks/               # Quarto tutorials
│   ├── 01_introduction.qmd
│   ├── 02_kalman_filter.qmd
│   └── 03_diagnostics.qmd
├── workflow/                # Snakemake pipeline
│   ├── Snakefile
│   └── config/params.yaml
├── docs/                    # Workshop materials
├── pyproject.toml           # Python packaging config
├── pixi.toml                # Pixi environment config
└── .pre-commit-config.yaml  # Code quality hooks
```

## Common Tasks

```bash
# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Check code style
pixi run lint

# Auto-fix style issues
pixi run lint-fix

# Format code
pixi run format

# Type check
pixi run typecheck

# Run all checks
pixi run check

# Install pre-commit hooks
pixi run hooks

# Render documentation
pixi run docs

# Run Snakemake workflow
pixi run workflow
```

## Workshop Materials

For the January 2026 workshop:

- **[TUTORIAL.md](docs/TUTORIAL.md)** - Step-by-step guide for participants
- **[WORKSHOP_GUIDE.md](docs/WORKSHOP_GUIDE.md)** - Instructor notes
- **[LEARNING_PATH.md](docs/LEARNING_PATH.md)** - 4-week progression

## Fork and Customize

This template is designed to be forked and customized:

1. **Fork** this repository
2. **Rename** `hive_protocol` to your project name
3. **Update** `pyproject.toml` and `pixi.toml` with your details
4. **Replace** the Kalman filter code with your domain logic
5. **Keep** the testing, CI/CD, and documentation patterns

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run `pixi run check` to ensure quality
5. Commit with a descriptive message
6. Push to your branch
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- CBG-ETH Zurich for supporting modern research practices
- The PyMC, Polars, Ruff, and Quarto communities
- All workshop participants who improve this template

---

*Built with modern Python for computational biology*
