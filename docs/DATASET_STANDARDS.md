# Dataset Standards (Sprint 20)

**Status:** v1.0 — testing infrastructure only, no algorithm code. Companion to [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) (fixture inventory, golden protocol) and [ARCHITECTURE.md](ARCHITECTURE.md) §10 (legal invariant: no copyrighted or externally-sourced imagery anywhere in this repo).

## 1. Purpose

`benchmarks/datasets/` is the permanent, categorized fixture set used throughout the project's lifetime for demos, docs, exploratory QA, and as a superset feeding the `benchmarks/perf` / `benchmarks/quality` / `tests/golden` suites. It complements — does not replace — the analytic-ground-truth fixtures in `benchmarks/framework/fixtures.py` (BENCHMARK_SPEC.md §2.1).

## 2. Categories

Eight fixed categories, declared in `benchmarks.datasets.metadata_schema.CATEGORIES`:

`animals`, `flowers`, `people`, `landscape`, `architecture`, `food`, `vehicles`, `cartoons`

Category membership is structural, not photographic: each category has a dedicated synthetic generator in `benchmarks/datasets/generators.py` that evokes the category through region shape/arrangement (e.g. `animals` → rounded blob silhouettes, `architecture` → rectilinear block grid, `landscape` → horizon bands). No photograph, scraped image, or external asset is ever used — this is a hard legal invariant (ARCHITECTURE.md §10), not a style choice.

## 3. Directory layout

```
benchmarks/datasets/
├── __init__.py
├── metadata_schema.py     # FixtureMetadata dataclass + CATEGORIES
├── generators.py          # one synthetic generator per category
├── registry.py            # fixture_id -> generator params + metadata (source of truth)
├── loaders.py             # public load_* API
├── examples/              # per-category metadata JSON for the small demo fixture
├── datasets/              # (reserved) any future on-disk cached artifacts
├── golden/                # GOLDEN_MANIFEST.json — frozen per-category subset
├── metadata/              # MANIFEST.json — content-hash manifest, every fixture
└── tests/                 # unit tests for generators/registry/loaders
```

Labels are generated in-memory, deterministically, from the registry entry (seed + params) — nothing pixel-heavy is committed to disk. What's on disk is metadata: manifests and per-fixture JSON, which are cheap to diff in review and regenerate.

## 4. Fixture id scheme

```
D-<category>-<tier>-<variant>
```

- `tier` is `examples` (one per category, 128×128, "easy") or `datasets` (the full ladder).
- `variant` is `01` for examples, or a difficulty (`easy`/`medium`/`hard`) for the datasets tier.

Examples: `D-animals-examples-01`, `D-architecture-datasets-hard`.

## 5. Metadata schema

Every fixture carries a `FixtureMetadata` record (`benchmarks/datasets/metadata_schema.py`):

| Field | Meaning |
|---|---|
| `fixture_id` | unique id, see §4 |
| `category` | one of the 8 categories |
| `width`, `height` | pixel resolution (also exposed as `resolution`, `megapixels`) |
| `difficulty` | `easy` / `medium` / `hard` |
| `palette_count` | declared color count for the fixture's label map |
| `expected_region_count` | declared approximate region count (analytic-style expectation, checked by dataset tests — not a QM gate) |
| `expected_printability` | declared printability proxy in `[0, 1]` |

Validation happens in `FixtureMetadata.__post_init__` — malformed metadata raises `ValueError` at registry-build time, not at test time.

## 6. Difficulty ladder

Each category has all three difficulties in the `datasets` tier:

| Difficulty | Resolution | Palette | Expected regions | Expected printability |
|---|---|---|---|---|
| easy | 256×256 | 6 | 10 | 0.85 |
| medium | 512×512 | 10 | 24 | 0.85 |
| hard | 768×768 | 16 | 48 | 0.70 |

## 7. Golden subset

`benchmarks/datasets/golden/GOLDEN_MANIFEST.json` freezes one fixture per category (the `datasets-medium` variant) by content hash, mirroring the golden-manifest shape used in `tests/golden/` (BENCHMARK_SPEC.md §4.1). Regenerate via the same script used to produce it (see §9) whenever a generator changes — never hand-edit the JSON.

## 8. Dataset versioning

`benchmarks.datasets.registry.DATASET_VERSION` follows the same rule as BENCHMARK_SPEC.md §2.2: adding a fixture bumps it; changing or removing an existing generator's output requires an ADR and manifest regeneration in the same commit.

## 9. Using the dataset

```python
from benchmarks.datasets.loaders import (
    load_fixture, load_examples, load_category, load_all, load_golden, dataset_manifest,
)

fx = load_fixture("D-animals-examples-01")
fx.labels        # np.int32 (H, W) label map
fx.metadata       # FixtureMetadata
fx.content_hash   # sha256 of the label map bytes

load_examples()   # 8 fixtures, one per category
load_category("architecture")  # every tier/difficulty for one category
load_all()        # every registered fixture
load_golden()      # frozen per-category golden subset
dataset_manifest()  # content-hash manifest matching metadata/MANIFEST.json
```

To regenerate the on-disk manifests after a generator or registry change:

```bash
python - <<'EOF'
import json
from pathlib import Path
from benchmarks.datasets.loaders import dataset_manifest, load_examples, load_golden
from benchmarks.datasets.registry import DATASET_VERSION

Path("benchmarks/datasets/metadata").mkdir(parents=True, exist_ok=True)
Path("benchmarks/datasets/metadata/MANIFEST.json").write_text(
    json.dumps(dataset_manifest(), indent=2, sort_keys=True) + "\n"
)

examples_dir = Path("benchmarks/datasets/examples")
examples_dir.mkdir(parents=True, exist_ok=True)
for fx in load_examples():
    (examples_dir / f"{fx.fixture_id}.json").write_text(
        json.dumps(fx.metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )

golden = {
    fx.fixture_id: {
        "category": fx.category, "content_hash": fx.content_hash,
        "shape": list(fx.labels.shape), **fx.metadata.to_dict(),
        "dataset_version": DATASET_VERSION,
    }
    for fx in load_golden()
}
Path("benchmarks/datasets/golden/GOLDEN_MANIFEST.json").write_text(
    json.dumps(golden, indent=2, sort_keys=True) + "\n"
)
EOF
```

## 10. Scope

This dataset is testing infrastructure: fixture generation, metadata, registry, and loaders only. It does not implement or modify any engine algorithm, stage, or validator — those remain governed by ENGINE_SPEC.md and the module design docs.
