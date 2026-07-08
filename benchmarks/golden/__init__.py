"""Golden-test framework over the Sprint 20 categorized dataset (Sprint 21).

Every fixture in ``benchmarks/datasets/`` is run through the real engine
pipeline (``benchmarks/framework/pipeline.run_pipeline``, no re-implementation
-- BENCHMARK_SPEC.md §6) to produce SVG + PDF-preview output, which is then
compared against a frozen golden using three independent checks:

- perceptual (PNG preview luminance SSIM)
- SVG structural (arc count, face-side multiset, per-arc segment counts)
- topology (region/arc/face count delta between golden and candidate)

All three reuse ``benchmarks/framework/visual.py``'s existing byte-hash +
structural-diff + SSIM machinery where possible; this package adds the
dataset-scoped runner, topology comparison, tolerance configuration, the
update ("bless") workflow, and report generation. No engine code is
touched -- this is testing infrastructure only.
"""

from __future__ import annotations
