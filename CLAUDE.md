# Claude Code Instructions for PyThermoNDT

## Foundation

**Read [AGENTS.md](AGENTS.md) first** - it contains project architecture, conventions, and patterns.

This file provides Claude-specific tool usage and workflow guidance.

## Tool Usage

### When to Use Which Tool

**Explore agent** - Open-ended codebase exploration:
- "How is RemoveFlash implemented?"
- "What test fixtures exist?"
- "Find transforms that modify DomainValues"

**Read tool** - Known specific files:
```
Read: src/pythermondt/transforms/sampling.py
Read: tests/conftest.py
```

**Glob tool** - Pattern matching:
```
Glob: src/pythermondt/transforms/*.py
Glob: tests/**/test_*.py
```

**Grep tool** - Content search:
```
Grep: "class.*Transform" in src/pythermondt/transforms/
Grep: "@pytest.fixture" in tests/
```

### Before Any Edit

**CRITICAL**: Always read files before editing them.

```
❌ BAD:  User: "Fix sampling.py" → Edit tool immediately
✅ GOOD: User: "Fix sampling.py" → Read sampling.py → Understand → Edit
```

### Parallel vs Sequential

**Parallel** (independent operations):
```python
# ✅ Read multiple files at once
Read(file1) + Read(file2) + Read(file3)

# ✅ Run independent checks
Bash(pytest) + Bash(ruff check) + Bash(mypy)
```

**Sequential** (dependent operations):
```python
# ✅ Edit before testing the edit
Edit(file.py) → Bash(pytest tests/)
```

## Common Workflows

### Adding a Transform

1. **Explore**: `Task(Explore, "Find similar transforms like UniformSampling")`
2. **Read**: Base class + similar transform
3. **Implement**: Follow pattern from AGENTS.md
4. **Verify**: `pytest tests/ && ruff check --fix .`

### Fixing a Bug

1. **Read**: Affected file + test file
2. **Test first**: Add failing test that reproduces bug
3. **Fix**: Minimal change to fix issue
4. **Verify**: Test passes, all tests pass, ruff passes

### Debugging Test Failures

```bash
# Run specific test with verbose output
pytest tests/path/test_file.py::test_name -vv

# Common issues:
# - Tensor shape mismatches (check H×W×T dimensions)
# - Missing DataContainer paths
# - Temporal inconsistency (Tdata vs DomainValues)
# - Unit mismatches after transforms
```

## Quick Command Reference

```bash
# Testing
pytest tests/                          # All tests
pytest -k "test_name"                 # Pattern match
pytest --benchmark-skip               # Skip benchmarks
pytest -x                             # Stop on first failure

# Quality
ruff check --fix . && ruff format .   # Lint and format
mypy src/pythermondt                  # Type check
pre-commit run --all-files            # All hooks

# Development
uv venv && uv pip install -e . && uv pip install -r requirements_dev.txt && pre-commit install
```

## Critical Patterns

### Temporal Consistency (Most Common Mistake)

**Always update Tdata and DomainValues together**:
```python
# ✅ CORRECT
new_tdata = tdata[..., indices]
new_domain = domain[indices] - domain[indices[0]]  # Zero-base!
container.update_datasets(
    ("/Data/Tdata", new_tdata),
    ("/MetaData/DomainValues", new_domain)
)

# ❌ WRONG - Only updating Tdata
container.update_dataset("/Data/Tdata", tdata[..., indices])
```

### Validate Early

```python
# ✅ At the start of forward()
if tdata.ndim != 3:
    raise ValueError(f"Expected 3D (H×W×T), got {tdata.shape}")
```

### Unit Updates

```python
# When transform changes physical meaning (e.g., ApplyLUT)
from pythermondt.data.units import Units
container.set_unit("/Data/Tdata", Units.KELVIN)
```

## Key File Locations

```
src/pythermondt/
├── transforms/
│   ├── base.py                # Base classes
│   ├── preprocessing.py       # ApplyLUT, RemoveFlash, CropFrames
│   ├── sampling.py            # Frame sampling
│   ├── normalization.py       # MinMax, Z-score
│   └── augmentation.py        # Data augmentation
├── readers/
│   ├── local_reader.py
│   └── s3_reader.py
├── data/datacontainer/
│   └── datacontainer.py       # Core container
└── config.py

tests/
├── conftest.py                # Global fixtures
├── integration/               # Integration tests
├── data/                      # Data module tests
├── dataset/                   # Dataset tests
└── io/                        # I/O tests (backends, parsers)
```

## When to Use EnterPlanMode

**Use for**:
- New features (transforms, readers)
- Multi-file changes
- Architectural changes
- Multiple valid approaches

**Skip for**:
- Typos, single-line fixes
- Simple test additions
- Obvious bug fixes

## Common Fixtures (from tests/conftest.py)

- `sample_tensor` - 3D tensor (96×96×100)
- `sample_container` - DataContainer with test data
- `localreader_with_file` - LocalReader with test file
- `sample_transform` - Example transform instance

## Essential Reminders

1. **Read before editing** - Never propose changes to unread code
2. **Maintain temporal consistency** - Update Tdata + DomainValues together
3. **Validate early** - Check shapes/dimensions at start of methods
4. **Test thoroughly** - Mirror structure, use parametrize
5. **Follow conventions** - Ruff (120 chars), Google docstrings, type hints
6. **Never mix** - Functionality and whitespace in same commit

See [AGENTS.md](AGENTS.md) for detailed architecture, patterns, and examples.
