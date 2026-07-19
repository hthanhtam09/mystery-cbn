"""Unit tests for foundation.codes (one-char cell code alphabet)."""

import pytest

from mysterycbn.foundation.codes import code_for_number


@pytest.mark.parametrize(
    ("number", "code"),
    [(1, "1"), (5, "5"), (9, "9"), (10, "0"), (11, "A"), (12, "B"), (17, "G"), (36, "Z")],
)
def test_mapping(number: int, code: str) -> None:
    assert code_for_number(number) == code


def test_all_codes_single_char_and_unique() -> None:
    codes = [code_for_number(n) for n in range(1, 37)]
    assert all(len(c) == 1 for c in codes)
    assert len(set(codes)) == 36


@pytest.mark.parametrize("number", [0, -1, 37, 100])
def test_out_of_range(number: int) -> None:
    with pytest.raises(ValueError):
        code_for_number(number)
