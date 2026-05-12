# PyThermoNDT Agent Instructions

## Style Baseline

Treat these as the canonical examples of the preferred style:

- `src/pythermondt/transforms/sampling.py` - `NonUniformSampling`
- `src/pythermondt/dataset/base_dataset.py` - `BaseDataset`

New code should feel like those implementations on the first pass:

- minimal, but not cryptic
- readable from top to bottom
- well documented, but only where documentation adds value
- explicit about validation, assumptions, and update steps

## Context File Scope

Keep this file intentionally concise.

- Include only non-obvious, high-value guidance that helps agents avoid mistakes in this repository.
- Prefer concrete rules and reference implementations over broad repository overviews.
- Do not restate information that is easy to discover from file names, README content, or normal code exploration.
- If a rule is not important enough to be followed on most tasks, omit it.

## Philosophy

- **Clarity over complexity**: Prefer direct control flow over clever abstractions.
- **Scientific rigor**: Preserve the physical meaning of thermal data and metadata.
- **PyTorch compatibility**: Keep transforms and datasets natural to use in PyTorch pipelines.
- **Smallest correct change**: Do not add helpers, layers, or backward-compatibility code without a concrete need.
- **No mixed commits**: Never mix functionality and whitespace-only cleanup in the same commit.
- **Tests for behavior changes**: All functional changes must include tests.

## Architecture

```text
Readers -> DataContainers -> Transforms -> Datasets -> PyTorch DataLoaders
```

Tech stack: Python 3.10-3.14, PyTorch >= 2.0, NumPy, h5py, boto3, pytest, ruff, mypy

### Standard DataContainer Paths

```text
/Data/Tdata                    # Thermal data (H x W x T)
/GroundTruth/DefectMask        # Defect mask (H x W)
/MetaData/LookUpTable          # Temperature conversion (uint16 -> float64)
/MetaData/DomainValues         # Domain values, usually time (T,)
/MetaData/ExcitationSignal     # Heating pattern (T,)
```

Use `container.get_datasets(...)` and `container.update_datasets(...)` when several paths move together.

## Coding Style

### Preferred Structure

Inside a method, prefer a simple sequence like this:

1. Load the relevant data.
2. Validate inputs and invariants early.
3. Compute the result with direct tensor operations.
4. Update the container or return the result.

That shape is preferred over deeply nested branches or many tiny helpers.

### Helpers

Extract a private helper only when it does at least one of these:

- names a real domain concept
- isolates dense math or interpolation logic
- is reused
- materially improves readability of the public method

`NonUniformSampling` is a good model: the helper methods isolate non-trivial math, while `forward()` stays linear and readable.

### Naming

- Use concrete domain names: `domain_values`, `excitation_signal`, `det_transforms`, `runtime_transforms`.
- Avoid vague names like `data2`, `temp`, `helper`, `manager2`, `process_data_wrapper`.
- Keep public APIs boring and obvious.

### Comments

- Use comments sparingly.
- Good comments explain why something is done or mark a phase boundary.
- Do not narrate obvious lines.

### Docstrings

- Use Google-style docstrings.
- Start with a short summary sentence.
- Document intent, important constraints, and non-obvious behavior.
- Document `Args`, `Returns`, and `Raises` when they add useful information.
- Avoid docstrings that merely restate the code.

### Errors and Validation

- Validate dimensions, bounds, modes, and units at the start of the relevant method.
- Raise descriptive exceptions with concrete context.

Example:

```python
if idx < 0 or idx >= len(self):
    raise IndexError(f"Index {idx} out of range [0, {len(self) - 1}].")
```

### Type Hints and Formatting

- Use modern Python 3.10+ type hints: `list[str]`, `dict[str, float] | None`
- Follow Ruff formatting and lint rules.
- Use double quotes.
- Keep lines within 120 characters.

## Project Patterns

### Transform Pattern

```python
class MyTransform(ThermoTransform):
    def __init__(self, param: int):
        super().__init__()
        self.param = param

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")

        if tdata.ndim != 3:
            raise ValueError(f"Expected 3D tensor (H x W x T), got {tdata.shape}.")

        processed = self._process(tdata)
        container.update_dataset("/Data/Tdata", processed)
        return container
```

Keep `forward()` easy to scan. If a transform contains dense math, isolate that math in private helpers and keep the main path linear.

### Dataset Pattern

`BaseDataset` is the reference for dataset code:

- keep `__getitem__()` direct and explicit
- validate indices before any load
- keep cache behavior readable and local
- use private state only where it clearly simplifies the public API

## Critical Domain Rules

### Temporal Consistency

Whenever frames are selected, resampled, or reordered, update the temporal metadata together:

```python
container.update_datasets(
    ("/Data/Tdata", new_tdata),
    ("/MetaData/DomainValues", new_domain_values),
    ("/MetaData/ExcitationSignal", new_excitation_signal),
)
```

If the new sequence starts at a later time, zero-base `DomainValues` when that is the established behavior of the transform.

### Unit Management

Update units when a transform changes physical meaning:

```python
from pythermondt.data.units import Units

container.set_unit("/Data/Tdata", Units.KELVIN)
```

### Performance

- Prefer tensor operations over Python loops.
- Flatten only when it clearly simplifies vectorized computation.
- Keep memory behavior understandable; do not hide expensive copies.
- Use `settings.num_workers` when worker count should follow project configuration.

## Testing

- All functional changes must include tests.
- Mirror the source layout under `tests/`.
- Use fixtures from `tests/conftest.py`.
- Prefer parametrized tests for shape, bounds, and mode coverage.
- For bug fixes, add or update a test that fails before the fix and passes after it.

Common locations:

- `tests/data/`
- `tests/dataset/`
- `tests/io/`
- `tests/integration/`

## Validation Commands

```bash
pytest tests/
pytest -k "test_name"
pytest --benchmark-skip
ruff check --fix .
ruff format .
mypy src/pythermondt
pre-commit run --all-files
```

Run the smallest relevant test set during development, then run the broader checks appropriate for the change.

## Key Locations

- Transforms: `src/pythermondt/transforms/{base,preprocessing,sampling,normalization,augmentation}.py`
- Datasets: `src/pythermondt/dataset/`
- Readers: `src/pythermondt/readers/{local_reader,s3_reader}.py`
- Tests: `tests/conftest.py`, `tests/{data,dataset,io,integration}/`
- Config: `pyproject.toml`, `src/pythermondt/config.py`

## Essential Rules

1. Match the coding style of `NonUniformSampling` and `BaseDataset`.
2. Prefer direct, linear implementations over extra abstraction.
3. Validate early and fail with clear, contextual errors.
4. Maintain temporal consistency across `Tdata`, `DomainValues`, and `ExcitationSignal`.
5. Update units when physical meaning changes.
6. Use tensor operations over loops when possible.
7. Include tests for every functional change.
8. Do not mix functionality and whitespace-only cleanup.

Contributions should feel native to the codebase: compact, readable, explicit, and scientifically correct.
