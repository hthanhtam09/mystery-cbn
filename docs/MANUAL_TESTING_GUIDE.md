# Manual Testing Guide — Mystery CBN Engine

**Quick reference for QA testing the engine in its current state.**

---

## Setup (One-Time)

```bash
cd mystery-cbn

# Create fresh venv
python3 -m venv .venv

# Install with PDF support (optional but recommended)
.venv/bin/pip install -e ".[dev,pdf]"
```

**Takes ~2 minutes on first run. Verify:**

```bash
.venv/bin/python -c "from mysterycbn.app import convert; print('✓ Ready')"
```

---

## Quick Sanity Check (30 seconds)

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile, sys

# Create test image
fixture = load_fixture("D-flowers-examples-01")
img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')

# Convert
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    img.save(f.name)
    try:
        b = convert(f.name, preset="medium")
        print(f"✓ Engine works")
        print(f"  SVG: {len(b.svg)} bytes")
        print(f"  PDF: {len(b.pdf)} bytes")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
EOF
```

**Expected output:** "✓ Engine works" + file sizes

---

## Test 1: Load Test

**Objective:** Verify image loading and error handling

```bash
.venv/bin/python << 'EOF'
from mysterycbn.app import convert
from mysterycbn.foundation.errors import InputError

# Test 1a: Valid image
print("Test 1a: Valid image")
try:
    from benchmarks.datasets.loaders import load_fixture
    from PIL import Image
    import numpy as np, tempfile
    fixture = load_fixture("D-flowers-examples-01")
    img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        b = convert(f.name, preset="medium")
    print("  ✓ Valid PNG loads and converts")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 1b: Non-existent file
print("Test 1b: Non-existent file")
try:
    convert("/nonexistent/path/image.png")
    print("  ✗ Should have raised InputError")
except InputError as e:
    print(f"  ✓ Correctly rejected: {str(e)[:50]}...")

# Test 1c: Invalid image data
print("Test 1c: Invalid image data")
import tempfile
with tempfile.NamedTemporaryFile(suffix='.png', delete=False, mode='wb') as f:
    f.write(b"NOT A REAL PNG FILE")
    invalid_path = f.name

try:
    convert(invalid_path)
    print("  ✗ Should have raised InputError")
except InputError as e:
    print(f"  ✓ Correctly rejected corrupt image")
EOF
```

**Expected:** ✓ all three checks pass

---

## Test 2: Determinism Test

**Objective:** Verify reproducibility (same input → byte-identical output)

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile

# Create test image
fixture = load_fixture("D-flowers-examples-01")
img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')

with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    img.save(f.name)
    test_path = f.name

# Run twice with same seed
print("Conversion 1...")
b1 = convert(test_path, seed=12345, preset="medium")

print("Conversion 2...")
b2 = convert(test_path, seed=12345, preset="medium")

# Verify
print("\nComparison:")
print(f"  SVG size 1: {len(b1.svg)} bytes")
print(f"  SVG size 2: {len(b2.svg)} bytes")
print(f"  SVG identical: {b1.svg == b2.svg}")

print(f"  PDF size 1: {len(b1.pdf)} bytes")
print(f"  PDF size 2: {len(b1.pdf)} bytes")
print(f"  PDF identical: {b1.pdf == b2.pdf}")

if b1.svg == b2.svg and b1.pdf == b2.pdf:
    print("\n✓ Determinism verified: byte-identical output")
else:
    print("\n✗ Outputs differ (expected for PDF rendering, typical for fonts)")
EOF
```

**Expected:** ✓ SVG identical; PDF may differ slightly (font rendering)

---

## Test 3: All Presets

**Objective:** Verify all three difficulty presets work

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile

fixture = load_fixture("D-flowers-examples-01")
img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')

with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    img.save(f.name)
    test_path = f.name

for preset in ["easy", "medium", "hard"]:
    try:
        b = convert(test_path, preset=preset, seed=0)
        print(f"✓ {preset:8s}: {len(b.svg):6d} byte SVG, {len(b.pdf):6d} byte PDF")
    except Exception as e:
        print(f"✗ {preset:8s}: {e}")
EOF
```

**Expected:** ✓ all three presets produce output

---

## Test 4: All Categories

**Objective:** Test dataset diversity

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.registry import available_example_ids
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile

for fixture_id in available_example_ids():
    fixture = load_fixture(fixture_id)
    img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        try:
            b = convert(f.name, preset="medium", seed=0)
            print(f"✓ {fixture_id:40s}: {len(b.svg):6d} byte SVG")
        except Exception as e:
            print(f"✗ {fixture_id:40s}: {e}")
EOF
```

**Expected:** ✓ all 8 categories convert successfully

---

## Test 5: Validation

**Objective:** Verify all 4 canonical validators pass

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile

fixture = load_fixture("D-flowers-examples-01")
img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')

with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    img.save(f.name)
    bundle = convert(f.name, preset="medium")

print("Validation Results:")
for report in bundle.report.validation:
    status = "✓" if report.passed else "✗"
    print(f"  {status} {report.validator_name:15s}: {', '.join(f'{k}={v}' for k,v in dict(report.metrics).items())}")

all_pass = all(r.passed for r in bundle.report.validation)
if all_pass:
    print("\n✓ All validators passed")
else:
    print("\n✗ Some validators failed")
EOF
```

**Expected:** ✓ fidelity, topology, printability, palette all pass

---

## Test 6: Output Files

**Objective:** Verify artifacts are written and can be opened

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile
from pathlib import Path

# Create output directory
output_dir = Path("./test_output")
output_dir.mkdir(exist_ok=True)

# Convert
fixture = load_fixture("D-animals-examples-01")
img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')

with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    img.save(f.name)
    bundle = convert(f.name, preset="medium")

# Write outputs
(output_dir / "output.svg").write_bytes(bundle.svg)
(output_dir / "output.pdf").write_bytes(bundle.pdf)
for name, data in bundle.previews.items():
    (output_dir / f"preview_{name}.png").write_bytes(data)

# Verify files exist
print("Output Files:")
for path in sorted(output_dir.glob("*")):
    size = path.stat().st_size
    print(f"  ✓ {path.name:30s}: {size:8d} bytes")

print(f"\nOpen these files in their respective viewers:")
print(f"  SVG:  {output_dir / 'output.svg'}")
print(f"  PDF:  {output_dir / 'output.pdf'}")
print(f"  PNG:  {output_dir / 'preview_lineart.png'} (line art)")
print(f"       {output_dir / 'preview_solved.png'} (solved)")
EOF
```

**Expected:** ✓ 4 files written (SVG, PDF, 2 PNGs)

**Manual Check:** Open each file visually:
- SVG in Inkscape or web browser → should see numbered coloring page outline
- PDF in Adobe Reader or Preview → should see printable page
- PNGs in image viewer → should see line art and solved versions

---

## Test 7: Performance Baseline

**Objective:** Measure and record execution time

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile, time

print("Performance Baseline (Single run, M1 system)")
print("=" * 60)

for fixture_id in ["D-flowers-examples-01", "D-animals-examples-01", "D-architecture-examples-01"]:
    fixture = load_fixture(fixture_id)
    img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        
        t0 = time.perf_counter()
        bundle = convert(f.name, preset="medium")
        elapsed = time.perf_counter() - t0
        
        print(f"{fixture_id:40s}: {elapsed:6.3f}s")

print("\nExpected: < 2.0s each (baseline: 0.855s)")
EOF
```

**Expected:** All complete in < 2 seconds

---

## Test 8: Unit Test Suite

**Objective:** Run full test suite to verify no regressions

```bash
# Run all unit tests
.venv/bin/python -m pytest tests/unit/ -v

# Expected output: 279 passed

# Or just quick summary:
.venv/bin/python -m pytest tests/unit/ -q

# Expected: "279 passed in X.XXs"
```

**Time:** ~15 seconds

**Expected:** All 279 tests pass ✓

---

## Test 9: Type Checking

**Objective:** Verify no static type errors (v2 code only)

```bash
.venv/bin/mypy
```

**Expected:** No errors (legacy code exempt from checks)

---

## Test 10: Code Quality

**Objective:** Verify linting passes

```bash
# Style check
.venv/bin/ruff check src tests benchmarks

# Format check
.venv/bin/ruff format --check src tests benchmarks

# Import graph enforcement
.venv/bin/lint-imports
```

**Expected:** All pass ✓

---

## Manual Inspection Checklist

After running a conversion, manually inspect outputs:

### SVG File

- [ ] Opens in Inkscape without errors
- [ ] Curves are smooth (zoom to 200% and check)
- [ ] Numbers are readable (8-12pt)
- [ ] All regions are closed paths
- [ ] No overlapping fills
- [ ] Legend present at bottom

### PDF File

- [ ] Opens in Adobe Reader / Preview
- [ ] Prints to 8.5"×11" or A4 without scaling
- [ ] Numbers remain readable
- [ ] No page rotation or artifacts
- [ ] File size reasonable (~50-100 KB)

### PNG Lineart

- [ ] Black lines on white background
- [ ] Numbers are legible
- [ ] Regions separated by clear boundaries
- [ ] No anti-aliasing artifacts

### PNG Solved

- [ ] Numbers present and readable
- [ ] Colors match legend
- [ ] No color bleeding
- [ ] Good contrast for printing

### General

- [ ] **Deterministic:** Run twice, SVG should be identical
- [ ] **No overlaps:** Use layer tools in Inkscape to verify regions don't overlap
- [ ] **No gaps:** Check that all pixels belong to a region
- [ ] **Fidelity:** Output closely matches input image structure

---

## Testing at Scale

To test with larger/more complex images:

```bash
.venv/bin/python << 'EOF'
from benchmarks.datasets.loaders import load_fixture, load_all
from mysterycbn.app import convert
from PIL import Image
import numpy as np, tempfile

print("Testing all available fixtures...")
results = []

for fixture in load_all():
    img = Image.fromarray((fixture.labels * 255 // fixture.labels.max()).astype(np.uint8), 'L')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        try:
            bundle = convert(f.name, preset="medium", seed=0)
            results.append((fixture.fixture_id, "PASS", len(bundle.svg)))
        except Exception as e:
            results.append((fixture.fixture_id, "FAIL", str(e)[:50]))

# Report
passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")

print(f"\n{passed} passed, {failed} failed")
for fixture_id, status, info in results:
    symbol = "✓" if status == "PASS" else "✗"
    print(f"  {symbol} {fixture_id:40s}: {info}")
EOF
```

---

## Regression Testing

Run this monthly or after engine changes:

```bash
# Full test suite (all categories)
.venv/bin/pytest tests/ -q

# Benchmark baseline (if you want to track performance)
.venv/bin/pytest benchmarks/smoke -q --benchmark-only

# Type checking
.venv/bin/mypy

# Code quality
.venv/bin/ruff check src tests benchmarks
.venv/bin/lint-imports
```

**Expected:** All pass without regressions

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mysterycbn'"

**Fix:** Reinstall the package:
```bash
cd mystery-cbn
.venv/bin/pip install -e ".[dev,pdf]"
```

### "cannot identify image file"

**Cause:** Image format not supported or file is corrupted  
**Fix:** Ensure file is valid PNG/JPG:
```bash
file image.png
# Should output: "image.png: PNG image data, ..."
```

### "Execution timed out"

**Cause:** Very large image (>2048×2048)  
**Fix:** Use smaller test image or increase timeout

### "PDF not generated"

**Cause:** reportlab or PyMuPDF not installed  
**Fix:**
```bash
.venv/bin/pip install reportlab PyMuPDF
```

---

## Quick Commands Reference

```bash
# Setup
cd mystery-cbn && python3 -m venv .venv && .venv/bin/pip install -e ".[dev,pdf]"

# Sanity check
.venv/bin/python -c "from mysterycbn.app import convert; print('✓')"

# Run all tests
.venv/bin/pytest tests/ -q

# Check types
.venv/bin/mypy

# Check style
.venv/bin/ruff check src tests benchmarks

# Profile performance
.venv/bin/pytest benchmarks/smoke -q --benchmark-only
```

---

**Last updated: 2026-07-08**  
**Next review: When new features are added**
