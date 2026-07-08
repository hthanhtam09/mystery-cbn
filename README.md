# mystery-cbn

Region-based mystery color-by-number conversion engine. Converts photos/illustrations into printable numbered-region coloring pages (SVG/PDF/PNG) without redrawing the source image.

- **Design:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — read this before touching any module.
- **Status:** end-to-end conversion implemented (Sprint 19, `docs/modules/orchestrator.md`). Not yet implemented: CLI/HTTP adapters, `palette_order` mystery-shuffle, optional plugins (`edge_snap`, `split_large`).

## Usage

```python
from mysterycbn.app import convert

bundle = convert("examples/flower.jpg", preset="medium")  # "easy" | "medium" | "hard"

bundle.svg                       # bytes
bundle.pdf                       # bytes | None
bundle.previews["lineart"]       # bytes (PNG)
bundle.previews["solved"]        # bytes (PNG)
bundle.report                    # RunReport: resolved_config, stage_timings_s, validation, ...
```

`convert()` is the engine's only public entry point (ARCHITECTURE.md §5). It is atomic: either every artifact validates and an `OutputBundle` is returned, or an `EngineError` subclass is raised and nothing partial is exposed.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest          # unit + property tests
.venv/bin/ruff check src tests benchmarks && .venv/bin/ruff format --check src tests benchmarks
.venv/bin/mypy                      # strict on v2 packages; legacy exempt until phase-3 migration
.venv/bin/lint-imports              # ARCHITECTURE.md §3 layer graph enforcement
.venv/bin/python -m pytest benchmarks/smoke -q   # benchmark harness smoke suite
```

CI (`.github/workflows/ci.yml`) runs the same five gates on every push; authoritative
benchmark numbers come from the pinned container in `benchmarks/Dockerfile`
(BENCHMARK_SPEC.md §9). Specs live in `docs/`; per-module designs in `docs/modules/`.
