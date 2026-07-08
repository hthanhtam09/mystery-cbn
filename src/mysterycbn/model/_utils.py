"""Shared helpers for the concrete data model (DATA_MODEL_SPEC.md §1).

Serialization regime: every object exposes ``to_dict()`` returning
JSON-compatible primitives. Bulk raster arrays serialize as metadata + SHA-256
(the debug-snapshot regime carries their payloads); geometric arrays serialize
in full. Internal serialization formats are not semver-governed.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def readonly(array: np.ndarray, dtype: type) -> np.ndarray:
    """Return a C-contiguous read-only copy of ``array`` (immutability rule)."""
    out: np.ndarray = np.ascontiguousarray(array, dtype=dtype)
    out.setflags(write=False)
    return out


def require(condition: bool, message: str) -> None:
    """Constructor validation helper: raise ``ValueError`` unless ``condition``."""
    if not condition:
        raise ValueError(message)


def require_hex64(value: str, field: str) -> None:
    """Validate a 64-char lowercase hex digest (provenance hashes)."""
    require(bool(_HEX64.match(value)), f"{field} must be 64 lowercase hex chars, got {value!r}")


def array_digest(array: np.ndarray) -> str:
    """SHA-256 of an array's raw bytes (shape/dtype-independent payload id)."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def array_meta(array: np.ndarray) -> dict[str, object]:
    """JSON-compatible metadata stand-in for a bulk array payload."""
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_digest(array),
    }
