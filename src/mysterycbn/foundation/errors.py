"""Engine exception hierarchy (ARCHITECTURE.md §11)."""

from __future__ import annotations


class EngineError(Exception):
    """Root of all engine-raised errors."""


class InputError(EngineError):
    """Unreadable, unsupported, or degenerate input. CLI exit 2 / HTTP 400."""


class ConfigError(EngineError):
    """Invalid or inconsistent configuration. CLI exit 3 / HTTP 422."""


class StageError(EngineError):
    """A pipeline stage failed; carries stage name and artifact state. CLI exit 5 / HTTP 500."""

    def __init__(self, message: str, *, stage_name: str, config_hash: str) -> None:
        super().__init__(message)
        self.stage_name = stage_name
        self.config_hash = config_hash


class QualityError(EngineError):
    """A validation gate failed and repair was impossible. CLI exit 4 / HTTP 409."""


class CancelledError(EngineError):
    """Cooperative cancellation was requested. CLI exit 130 / HTTP 499."""
