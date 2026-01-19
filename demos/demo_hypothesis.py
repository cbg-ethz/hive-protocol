"""Demo: Hypothesis property-based testing.

Run with: pixi run pytest demos/demo_hypothesis.py -v
"""

from hypothesis import given, settings, strategies as st


def remove_duplicates(items: list[int]) -> list[int]:
    """Remove duplicates while preserving order."""
    seen: set[int] = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


# Property-based test - Hypothesis generates hundreds of test cases
@given(st.lists(st.integers(min_value=-1000, max_value=1000)))
@settings(max_examples=200)
def test_remove_duplicates_no_duplicates(items: list[int]) -> None:
    """Property: result has no duplicates."""
    result = remove_duplicates(items)
    assert len(result) == len(set(result)), "Found duplicates in result!"


@given(st.lists(st.integers(min_value=-1000, max_value=1000)))
@settings(max_examples=200)
def test_remove_duplicates_preserves_elements(items: list[int]) -> None:
    """Property: all unique elements are preserved."""
    result = remove_duplicates(items)
    assert set(result) == set(items), "Lost some elements!"


@given(st.lists(st.integers(min_value=-1000, max_value=1000)))
@settings(max_examples=200)
def test_remove_duplicates_preserves_order(items: list[int]) -> None:
    """Property: relative order of first occurrences is preserved."""
    result = remove_duplicates(items)
    # Get first occurrence indices in original list
    first_indices = {x: i for i, x in reversed(list(enumerate(items)))}
    for i, x in enumerate(result):
        for j, y in enumerate(result[i + 1 :], i + 1):
            assert first_indices[x] < first_indices[y], f"Order violated: {x} should come before {y}"


if __name__ == "__main__":
    print("Run with: pixi run pytest demos/demo_hypothesis.py -v")
    print("\nHypothesis will generate hundreds of random test cases!")
