"""Cooperative cancellation (ARCHITECTURE.md §4.2, §11)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CancelToken(Protocol):
    """Checked by the kernel between stages; raising happens in the kernel, not here."""

    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        ...


class ManualCancelToken:
    """Default token: flipped once by the requesting side, polled by the kernel.

    Thread-safe by virtue of being a single monotonic bool flip (no
    read-modify-write races are possible).
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation; idempotent."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled
