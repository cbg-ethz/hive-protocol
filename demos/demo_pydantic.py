"""Demo: Pydantic configuration validation.

Run with: python demos/demo_pydantic.py
"""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""

    n_samples: int = Field(ge=1, le=10000, description="Number of MCMC samples")
    learning_rate: float = Field(gt=0, lt=1, description="Learning rate")
    output_dir: Path = Field(default=Path("results"))
    random_seed: int | None = Field(default=None)


def main() -> None:
    print("=" * 60)
    print("PYDANTIC VALIDATION DEMO")
    print("=" * 60)

    # Valid configuration
    print("\n1. Valid config:")
    config = ExperimentConfig(n_samples=1000, learning_rate=0.01)
    print(f"   {config}")
    print(f"   JSON: {config.model_dump_json()}")

    # Type coercion
    print("\n2. Type coercion (string '500' -> int 500):")
    config2 = ExperimentConfig(n_samples="500", learning_rate=0.1)  # type: ignore
    print(f"   n_samples type: {type(config2.n_samples).__name__} = {config2.n_samples}")

    # Invalid configuration - clear errors!
    print("\n3. Invalid config (n_samples=-5, learning_rate=2.0):")
    try:
        ExperimentConfig(n_samples=-5, learning_rate=2.0)
    except ValidationError as e:
        for error in e.errors():
            print(f"   - {error['loc'][0]}: {error['msg']}")

    print("\n" + "=" * 60)
    print("Your experiments fail fast with clear messages!")
    print("=" * 60)


if __name__ == "__main__":
    main()
