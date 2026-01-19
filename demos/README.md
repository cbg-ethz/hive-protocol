# Workshop Demos

Quick demos for the Hive Protocol workshop.

## Demo 1: Ruff Auto-Fix

```bash
# Show the messy code
cat demos/messy_code.py

# Fix it!
pixi run ruff check --fix demos/messy_code.py
pixi run ruff format demos/messy_code.py

# See the clean result
cat demos/messy_code.py

# Reset for next demo
git checkout demos/messy_code.py
```

## Demo 2: Pydantic Validation

```bash
python demos/demo_pydantic.py
```

Shows:
- Valid config creation
- Automatic type coercion
- Clear validation error messages

## Demo 3: Hypothesis Testing

```bash
pixi run pytest demos/demo_hypothesis.py -v
```

Shows Hypothesis generating hundreds of test cases automatically.

## Demo 4: Quarto Live Edit

```bash
quarto preview notebooks/01_introduction.qmd
```

Edit the file and watch it update live!
