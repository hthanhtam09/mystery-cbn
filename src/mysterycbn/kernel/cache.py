"""Content-addressed artifact caching interfaces (ARCHITECTURE.md §13.4).

Artifacts are keyed by (seed, source, upstream-chain) so that re-running with
a changed downstream knob never re-runs upstream stages, while any upstream
change — stage version, config section, order — invalidates everything after
it. Falls out of artifact immutability.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping

from mysterycbn.model.artifacts import Artifact


def section_hash(section: Mapping[str, object]) -> str:
    """Stable hash of one stage's config section (canonical JSON)."""
    payload = json.dumps(section, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chain_hash(upstream: str, stage_name: str, stage_version: str, cfg_hash: str) -> str:
    """Fold one stage's identity + config into the upstream chain hash."""
    payload = f"{upstream}\x1f{stage_name}\x1f{stage_version}\x1f{cfg_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_cache_key(seed: int, source_hash: str, upstream_chain: str) -> str:
    """The cache key for one stage's provided artifacts."""
    payload = f"{seed}\x1f{source_hash}\x1f{upstream_chain}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ArtifactCache(ABC):
    """Storage interface for stage outputs; implementations must be safe to
    miss (a cache is never load-bearing for correctness — I2 requires the
    cached path and the recomputed path to yield identical artifacts)."""

    @abstractmethod
    def get(self, key: str, artifact_name: str) -> Artifact | None:
        """Return the cached artifact or ``None`` on miss."""

    @abstractmethod
    def put(self, key: str, artifact_name: str, artifact: Artifact) -> None:
        """Store one provided artifact under the stage's cache key."""


class InMemoryArtifactCache(ArtifactCache):
    """Process-local cache (tests, single-run reuse). Unbounded by design —
    eviction policy is an adapter concern, not a kernel one."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Artifact] = {}

    def get(self, key: str, artifact_name: str) -> Artifact | None:
        return self._store.get((key, artifact_name))

    def put(self, key: str, artifact_name: str, artifact: Artifact) -> None:
        self._store[(key, artifact_name)] = artifact

    def __len__(self) -> int:
        return len(self._store)
