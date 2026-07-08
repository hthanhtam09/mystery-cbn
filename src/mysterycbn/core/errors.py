"""Engine exception hierarchy.

Every failure raised by the engine derives from :class:`EngineError` so the
API layer can map them to structured HTTP responses without catching bare
exceptions.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for all engine failures."""


class InputError(EngineError):
    """The input image is missing, unreadable, or unsupported."""


class ConfigError(EngineError):
    """The engine configuration is invalid or inconsistent."""


class StageError(EngineError):
    """A pipeline stage failed. Carries the stage name for diagnostics."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class ValidationError(EngineError):
    """A quality gate failed (topology, printability, palette)."""
