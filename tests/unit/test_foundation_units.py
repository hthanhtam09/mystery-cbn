"""Unit tests for foundation/units."""

from __future__ import annotations

import pytest

from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH, PageUnits


def test_mm_pt_known_values() -> None:
    units = PageUnits(work_scale=1.0)
    assert units.mm_to_pt(MM_PER_INCH) == pytest.approx(PT_PER_INCH)
    assert units.pt_to_mm(PT_PER_INCH) == pytest.approx(MM_PER_INCH)


def test_roundtrips() -> None:
    units = PageUnits(work_scale=0.37)
    for value in (0.0, 1.0, 3.5, 215.9):
        assert units.pt_to_mm(units.mm_to_pt(value)) == pytest.approx(value)
        assert units.pt_to_px(units.px_to_pt(value)) == pytest.approx(value)


def test_px_pt_uses_work_scale() -> None:
    units = PageUnits(work_scale=0.5)
    assert units.px_to_pt(10.0) == pytest.approx(5.0)
    assert units.pt_to_px(5.0) == pytest.approx(10.0)
    assert units.work_scale == 0.5


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_work_scale_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        PageUnits(work_scale=bad)
