"""Structured, correlation-scoped logging contract (ARCHITECTURE.md §12)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod


class RunLoggerFactory(ABC):
    """Produces per-module loggers stamped with a run-scoped correlation id.

    The library never configures global handlers; adapters do.
    """

    @abstractmethod
    def get_logger(
        self, module_namespace: str, correlation_id: str
    ) -> logging.LoggerAdapter[logging.Logger]:
        """Return a logger for ``module_namespace`` whose records carry ``correlation_id``."""


class CorrelatedLoggerFactory(RunLoggerFactory):
    """Default factory: stdlib loggers with the correlation id injected on every record.

    Namespaces are rooted at ``mysterycbn`` (§12: one logger per module namespace);
    the correlation id is exposed to formatters as ``record.correlation_id``.
    """

    _ROOT = "mysterycbn"

    def get_logger(
        self, module_namespace: str, correlation_id: str
    ) -> logging.LoggerAdapter[logging.Logger]:
        if not module_namespace.startswith(self._ROOT):
            module_namespace = f"{self._ROOT}.{module_namespace}"
        base = logging.getLogger(module_namespace)
        return logging.LoggerAdapter(base, extra={"correlation_id": correlation_id})
