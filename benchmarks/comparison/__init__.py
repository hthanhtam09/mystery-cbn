"""Technical quality comparison framework (Sprint 24): compares engine
output across configurations (difficulty presets today) using only
synthetic, in-repo evaluation samples -- never copyrighted or externally
sourced imagery (ARCHITECTURE.md §10 legal invariant; see
docs/TECHNICAL_QUALITY_COMPARISON.md).

The goal is comparing *technical* quality (region count, compactness,
boundary smoothness, average edge length, region size distribution, label
density, printability) across configurations -- not judging or reproducing
any particular artwork. No engine, stage, or rendering code is modified;
this package only measures and reports.
"""

from __future__ import annotations
