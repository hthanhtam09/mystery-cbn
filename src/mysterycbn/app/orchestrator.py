"""Orchestrator: plan → execute → validate → render → atomic OutputBundle (ARCHITECTURE.md §4.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mysterycbn.app.jobs import JobSpec
from mysterycbn.kernel.cancellation import CancelToken
from mysterycbn.kernel.progress import ProgressListener

# The concrete, validated dataclass (DATA_MODEL_SPEC §19: atomicity checks,
# exactly 4 embedded validation reports) -- not the looser structural
# Protocol of the same name in model/artifacts.py (field previews_png vs.
# previews; no atomicity/validator-count invariants). Only this one is ever
# actually constructible as a real return value (Sprint 19).
from mysterycbn.model.reports import OutputBundle


class Orchestrator(ABC):
    """The engine's single entry point for adapters — public interface #1 (ARCHITECTURE.md §5).

    Anything an adapter can do, a test can do.
    """

    @abstractmethod
    def convert(
        self,
        spec: JobSpec,
        *,
        on_progress: ProgressListener | None,
        cancel_token: CancelToken | None,
    ) -> OutputBundle:
        """Run the full pipeline. Atomic: either all artifacts validated and returned, or an
        EngineError subclass is raised and nothing is written."""
