"""One-character cell codes for printed palette numbers.

Commercial "mystery" color-by-number pages keep every cell code to a single
glyph so the code fits tiny camouflage cells: palette numbers 1-9 print as
digits, 10 prints as "0", and 11+ continue through the alphabet ("A", "B",
...). The data model keeps ``printed_number`` as a 1-based int everywhere
(model/layout.py); this mapping is applied only at text-emission boundaries
(labels font-fit, svg/pdf/png renderers, legend chips) so all of them agree.
"""

from __future__ import annotations

_MAX_CODE_NUMBER = 36  # 9 digits + "0" + 26 letters


def code_for_number(printed_number: int) -> str:
    """Map a 1-based printed palette number to its one-char cell code.

    1-9 -> "1".."9", 10 -> "0", 11-36 -> "A".."Z".
    """
    if not 1 <= printed_number <= _MAX_CODE_NUMBER:
        raise ValueError(
            f"printed_number must be in [1, {_MAX_CODE_NUMBER}], got {printed_number!r}"
        )
    if printed_number <= 9:
        return str(printed_number)
    if printed_number == 10:
        return "0"
    return chr(ord("A") + printed_number - 11)
