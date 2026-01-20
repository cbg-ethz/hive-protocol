# Modern Python for Computational Biology - Presenter Notes

This document contains spoken narration for each slide in the workshop presentation. Use for:
- Practicing the presentation
- Providing written materials to attendees
- Creating video voiceovers

---

## Title Slide
**Duration:** 30 seconds

> Welcome to "Modern Python for Computational Biology" - what we call the Hive Protocol. I'm [Name] from CBG at ETH Zurich. Over the next hour, we'll transform how you write Python code for research.

---

## SECTION: Foundations

### Hook Slide
**Duration:** 30 seconds

> What if every project you started already had tests, documentation, and CI working from day one? That's exactly what this template gives you. Fork this repo. Delete the example code. Ship research software.

### The Problem
**Duration:** 2 minutes

> Let's start with something we've all experienced. On the left, you see the typical state of research code: requirements.txt files with conflicting versions, no automated tests - because "I tested it manually" - no type hints, the classic "works on my machine" excuse, and Jupyter notebooks that are impossible to merge in git.

> The cost? Hours debugging environment issues, results that can't be reproduced, code that only the author understands, and papers with broken supplementary code. This isn't about being pedantic - it's about scientific integrity and your own sanity.

### The 2026 Landscape
**Duration:** 2 minutes

> The good news: The Python ecosystem has matured dramatically. These aren't experimental tools anymore - they're production-ready and widely adopted.

> Let's look at what we're replacing. [Point to Old Way column - shown with strikethrough]
> - pip and requirements.txt for package management
> - The Black-flake8-isort trio for code quality
> - pandas for everything data-related
> - Jupyter notebooks for literate programming
> - Manual test cases that only test what we thought of
> - And manual config validation that we often forget

> [Click to reveal Modern Way column - each tool appears]
> Now here's the 2026 stack:
> - **Pixi** - 10-100x faster, handles both conda and PyPI
> - **Ruff** - one tool, 30-100x faster
> - **Polars** - 5-50x speedups for large data
> - **Quarto** - clean git diffs, reproducible
> - **pytest + Hypothesis** - finds bugs you didn't know existed
> - **Pydantic** - automatic validation with clear errors

> All of these are included in this template and ready to use.

### Our Toolkit
**Duration:** 30 seconds

> Here's a visual overview of the tools we'll be using today. You don't need to memorize all of these - the template has everything configured and ready to go.

### Why Pixi?
**Duration:** 3 minutes

> So why did we choose Pixi specifically? In bioinformatics, we can't live without conda packages - samtools, bedtools, bcftools - these are conda-only.

> [Point to alternatives table]

> Conda is the original but it's slow and doesn't have lockfiles. Poetry is great but doesn't support conda. UV is blazing fast but also PyPI-only. Pixi gives us the best of all worlds: conda packages, PyPI packages, incredible speed, and deterministic lockfiles.

> The workflow is simple: `pixi install` sets everything up in about 30 seconds. Then `pixi run test`, `pixi run lint` - everything is a simple command.

### Pixi: Community Support
**Duration:** 1 minute

> And Pixi isn't just some niche tool - it's endorsed by both conda-forge and Anaconda. Look at this star history - it's growing rapidly because the community has recognized it solves a real problem.

> Important note: Pixi is completely free for research and all other uses. It's BSD3 licensed with no restrictions on academic or commercial use. The entire toolchain is open source.

### Why Ruff?
**Duration:** 2 minutes

> Ruff is a game-changer. One tool replaces Black for formatting, flake8 for linting, isort for import sorting, and pyupgrade for modernizing syntax. It has over 700 lint rules.

> The key? It's written in Rust, so it's 30-100x faster than the Python equivalents. On a large codebase, the difference between waiting 30 seconds and 300 milliseconds is huge for your workflow.

### Repository Structure
**Duration:** 1 minute

> Here's how the repository is organized. We use the `src` layout - your code goes in `src/hive_protocol`. Tests are separate in `tests/`. Quarto notebooks live in `notebooks/`.

> The key principle: `pyproject.toml` is the single source of truth. All your dependencies, tool configurations, build settings - everything lives in one file.

### Hands-on: Setup
**Duration:** 5 minutes (activity)

> Let's get everyone set up. Clone the repository, run pixi install, and verify with pixi run test. Raise your hand if you run into any issues.

---

## SECTION: Example Content - Bayesian Kalman Filter

### The Example: State-Space Model
**Duration:** 2 minutes

> The scientific content in this template is a Bayesian Kalman filter using PyMC. This is just example content - you'll replace it with your own domain logic.

> The key point here is: always check your diagnostics. R-hat under 1.01, ESS over 400, zero divergences. If you don't check diagnostics, you don't have results.

---

## SECTION: Testing & Type Safety

### Type Checking: mypy vs Pyright
**Duration:** 2 minutes

> Type hints are one of the biggest improvements to Python in the last decade. Look at the difference - without types, you have no idea what this function takes or returns. With types, it's crystal clear.

> We use two type checkers: Pyright for local development because it's fast, and mypy for CI because it's stable.

### Pyright: Fast Local Feedback
**Duration:** 1 minute

> Pyright is written in TypeScript and checks large codebases in seconds. If you use VS Code with Pylance, you already have Pyright. Real-time feedback as you type.

### mypy: Stable CI Checks
**Duration:** 1 minute

> mypy is the industry standard. It's battle-tested, has an extensive plugin ecosystem, and integrates well with pre-commit hooks. Every commit gets checked automatically.

### Why Pydantic?
**Duration:** 2 minutes

> Pydantic solves the configuration problem. Look at the left side - manual validation is tedious, repetitive, and error-prone. You have to check every field, every type, every constraint.

> With Pydantic, you define your model once, and validation happens automatically. Type coercion, constraint checking, serialization - all built in.

### Pydantic for Scientific Configs
**Duration:** 2 minutes

> Here's what a real scientific configuration looks like with Pydantic. You define your simulation parameters with constraints - n_steps must be between 10 and 10000, noise values must be positive.

> The benefits are huge: strings become integers automatically, invalid values are caught immediately, and you can serialize to JSON or load from YAML with one line.

### Pydantic + Type Checkers
**Duration:** 1 minute

> The magic happens when you combine Pydantic with type checkers. mypy and Pyright catch errors at development time. Pydantic catches validation errors at runtime. Defense in depth.

### Hypothesis: Property-Based Testing
**Duration:** 2 minutes

> Hypothesis is different from normal testing. Instead of you picking test examples, Hypothesis generates hundreds of random inputs automatically.

> Look at the difference: with example-based tests, you only test what you think of. With property-based tests, Hypothesis finds edge cases you never considered.

### Hypothesis Strategies
**Duration:** 1 minute

> Hypothesis has built-in strategies for all common types - integers, floats, text, lists, dictionaries. And for scientific computing, there are NumPy array strategies.

### Hypothesis in Practice
**Duration:** 1 minute

> Here's a real example from the template. We test that the Kalman filter has certain properties regardless of the input parameters. Observations should be noisier than true states. Lengths must match.

> Key insight: test properties that must always hold, not specific values.

---

## SECTION: Workflows & Quality

### Git for Researchers
**Duration:** 2 minutes

> Use feature branches. Never commit directly to main. Create a branch, make your changes, push, and open a pull request.

> And please: write good commit messages. "fix" tells us nothing. "fix: correct off-by-one error in filter" tells the whole story.

### Pre-commit Hooks
**Duration:** 1 minute

> Pre-commit hooks are your safety net. When you run git commit, Ruff and mypy run automatically. If there are issues, the commit is blocked. You fix the issues and commit again.

> Only clean code reaches the repository.

### GitHub Actions CI
**Duration:** 1 minute

> Every push triggers automated testing on GitHub Actions. Tests run, linting runs, type checking runs. If anything fails, you see it immediately.

> That green badge on your README? Instant trust.

### CI: Resource-Efficient Testing
**Duration:** 30 seconds

> Look at this dependency graph. Jobs run in parallel where possible - lint and typecheck don't need to wait for each other. Tests run on multiple platforms simultaneously. Fast feedback, minimal compute waste.

### CI: This is Trust
**Duration:** 30 seconds

> When you see all those green checks passing, you know your code works. Add features without fear. Refactor without anxiety. The CI has your back - it catches problems before they reach main.

### Quality Isn't Optional. It's Automated.
**Duration:** 1 minute

> Here's what runs automatically on every push. On the left: pytest-cov showing 97% test coverage. On the right: interrogate showing 94% docstring coverage.

> The key message: low friction, high standards. You write good code, the tools verify it automatically. Quality isn't something you do manually - it's enforced by the system.

### The Development Loop
**Duration:** 1 minute

> Here's the workflow: branch, code, test, commit, push, PR, review, merge. Rinse and repeat. The tools make each step fast and reliable.

### Fork and Customize
**Duration:** 1 minute

> This template is designed to be forked. Change the name, update the metadata, replace the Kalman filter with your code. Keep the testing patterns, keep the CI, keep the pre-commit hooks.

### Hands-on: Make a Change
**Duration:** 5 minutes (activity)

> Let's try it. Create a branch, make a small change - add a docstring, fix a typo - run the checks, and commit. Watch pre-commit run.

---

## SECTION: Hands-On Demos

### Demo: Ruff Auto-Fix
**Duration:** 3 minutes

> Watch what Ruff does. I have this messy code - unused imports, bad spacing, unused variables. Two commands: ruff check --fix and ruff format. Done. Clean code.

### Demo: Quarto Live Edit
**Duration:** 3 minutes

> Start the preview server with quarto preview. Now every time you save the file, it re-renders instantly. Code, markdown, output - all in one git-friendly file.

### Demo: Pydantic Validation
**Duration:** 3 minutes

> Watch what happens when I pass invalid values. Pydantic catches them immediately with clear error messages. n_samples should be >= 1. learning_rate should be < 1. Your experiments fail fast with clear messages.

### Demo: Hypothesis Bug Finding
**Duration:** 3 minutes

> Run pytest with this test. Hypothesis generates hundreds of test cases. When it finds a failure, it shrinks the input to the minimal failing example. That's how you find bugs you didn't know existed.

---

## SECTION: Summary

### What We Covered
**Duration:** 1 minute

> Quick recap: Pixi and Ruff for speed, Pyright and mypy for type safety, Pydantic for configuration validation, Hypothesis for property-based testing, and CI/CD for automation.

### Resources
**Duration:** 30 seconds

> Everything is linked here. The repository, documentation for each tool. Take a photo or check the slides later.

### Next Steps
**Duration:** 1 minute

> Today: fork the repository. This week: adapt it to your project. Ongoing: share with your lab.

> Questions?

---

## TIMING SUMMARY

| Section | Slides | Duration |
|---------|--------|----------|
| Foundations | 8 | 15 min |
| Kalman Example | 1 | 2 min |
| Testing & Types | 9 | 15 min |
| Workflows | 7 | 12 min |
| Hands-On Demos | 4 | 12 min |
| Summary | 3 | 4 min |
| **Total** | **32** | **60 min** |

---

## Q&A PREPARATION

### Common Questions

**Q: Why not just use conda?**
> Conda is slow and doesn't have lockfiles. Pixi uses the same package ecosystem but adds speed, lockfiles, and task management. It's a superset of conda functionality.

**Q: Is Polars production-ready?**
> Absolutely. It's used by major companies and has a stable 1.0 release. The API is mature and well-documented.

**Q: Do I need both mypy and Pyright?**
> Not necessarily. We recommend Pyright for local development (faster feedback) and mypy for CI (more stable). But using just one is fine.

**Q: How do I migrate an existing project?**
> Start with Ruff - it's the easiest win. Then add type hints gradually. Pixi can import from requirements.txt. Don't try to do everything at once.

**Q: Is Pydantic overkill for small scripts?**
> For tiny scripts, maybe. But the moment you have a config file or command-line arguments, Pydantic pays for itself immediately.

---

## DEMO CHECKPOINTS

Before the workshop, verify:

- [ ] `pixi install` completes successfully
- [ ] `pixi run test` passes
- [ ] `pixi run lint` shows no errors
- [ ] Quarto renders slides correctly
- [ ] All demo code snippets work in `pixi shell`

---

## BACKUP PLANS

**If pixi install fails:**
Use pre-built environment or have attendees pair up with someone who has it working.

**If network is slow:**
All examples work offline after initial setup. Focus on pre-cloned repos.

**If PyMC sampling is too slow:**
Reduce `n_samples` to 100, `n_tune` to 50.

**If VS Code isn't available:**
All commands work from terminal. Pyright can run standalone.
