# PyThermoNDT Agent Instructions

## Code Style

Use these implementations as style references:

- `src/pythermondt/transforms/sampling.py` - `NonUniformSampling`
- `src/pythermondt/dataset/base_dataset.py` - `BaseDataset`

Read the code you will change and a relevant reference before editing. Follow the surrounding style unless it conflicts
with an explicit rule here. Use the formatter and lint settings in `pyproject.toml`.

### Structure and Naming

- Write compact code that is easy to read from top to bottom. Fewer lines are useful only when they remain clear.
- Keep the main path direct: load data, validate required invariants, compute, then update or return.
- Use early errors instead of deep nesting. Validate inputs before expensive work or state changes; do not repeat checks
  already guaranteed by the calling code or container API.
- Raise descriptive errors with the invalid value and expected bounds, shape, mode, or unit when useful.
- Name variables by their domain meaning, such as `domain_values`, `excitation_signal`, and `runtime_transforms`.
- Use modern Python type annotations and explicit signatures, compatible with the supported versions in `pyproject.toml`.

Extract a private helper when it isolates dense math, names a domain operation, is reused, or materially improves
readability. Do not split a clear sequence into many one-line helpers or add layers only to make code look organized.
`NonUniformSampling` shows how to isolate interpolation and sampling math while keeping `forward()` direct.

### Documentation

- Use short Google-style docstrings for public APIs and non-obvious behavior.
- Document intent, assumptions, constraints, and physical meaning. Add `Args`, `Returns`, and `Raises` when useful.
- Use comments to explain why an operation is needed or mark a meaningful phase. Do not narrate obvious code.

### Tensor Operations and Memory

- Prefer direct tensor or bulk operations over per-element Python loops.
- Flatten tensors only when it makes the computation clearer. Keep copies and large allocations visible.
- Keep transforms and datasets compatible with PyTorch pipelines.
- Use `settings.num_workers` when the worker count should follow project configuration.

## Data and Transform Invariants

Standard container paths and shapes:

```text
/Data/Tdata                    # Thermal data (H x W x T)
/GroundTruth/DefectMask         # Defect mask (H x W)
/MetaData/LookUpTable           # Temperature conversion (uint16 -> float64)
/MetaData/DomainValues          # Domain values, usually time (T,)
/MetaData/ExcitationSignal      # Heating pattern (T,), when present
```

- Use `container.get_datasets(...)` and `container.update_datasets(...)` when related paths move together.
- When selecting, resampling, or reordering frames, keep `Tdata`, `DomainValues`, and any present `ExcitationSignal`
  aligned. Do not make optional metadata required unless the operation needs it.
- Preserve the domain origin unless the transform defines a reset. Zero-base time values when that is the established
  behavior; do not apply time-specific changes to another domain.
- Update dataset units when an operation changes physical meaning. Use `Units` and `container.set_unit(...)`.
- Preserve the physical meaning of thermal values and metadata, not only tensor shapes.

For dataset changes, keep `__getitem__()` direct and validate indices before loading. Preserve cache isolation: runtime
transforms must not modify cached data. Cache only the deterministic prefix of a transform chain; the first random
transform and all later transforms must run at runtime. Keep cache ownership and worker behavior explicit.

## Tests and Validation

- Every functional change must include tests for meaningful changed behavior. Update existing tests when sufficient.
- For bug fixes, add or update a test that fails before the fix and passes after it.
- Follow the matching test module and reuse fixtures from `tests/conftest.py` when appropriate.
- Use parametrized tests for meaningful shape, bounds, and mode coverage. Avoid redundant smoke tests and tests tied
  only to private implementation details.
- Run the smallest relevant checks first, then broader checks appropriate to the change. Report checks you could not run.

Common commands, with paths narrowed to the affected files or tests:

```bash
pytest tests/<affected_test_module>.py --benchmark-skip
ruff check <changed_python_files>
ruff format --check <changed_python_files>
mypy src/pythermondt
pre-commit run --files <changed_files>
```

For broader behavior changes, run `pytest tests/ --benchmark-skip`. Apply formatting and auto-fixes only to relevant files;
do not use whole-repository fixes as routine validation.

## Change Boundaries

- Make the smallest correct change. Preserve existing behavior unless the task requires a change.
- Do not add speculative abstractions or backward-compatibility code without a concrete need.
- Do not perform unrelated cleanup or mix functional changes with unrelated whitespace changes.
- Do not commit unless explicitly requested.

Keep this file self-contained and concise. Keep each rule in one place, and prefer project conventions and domain
invariants over directory inventories or information already defined in tool configuration.
