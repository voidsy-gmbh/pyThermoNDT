# Claude Code Instructions for PyThermoNDT

Read `AGENTS.md` first. It defines the project architecture, domain rules, and the preferred coding style.

Keep this file intentionally concise.

- Include only non-obvious, high-value guidance.
- Prefer concrete rules over repository overviews.
- Do not duplicate documentation that is easy to discover elsewhere in the repo.

## Canonical Style References

Match these implementations unless the surrounding file clearly uses a different local pattern:

- `src/pythermondt/transforms/sampling.py` - `NonUniformSampling`
- `src/pythermondt/dataset/base_dataset.py` - `BaseDataset`

Target that style on the first try:

- minimal and direct
- readable top-to-bottom
- helpful docstrings, not verbose docstrings
- sparse comments that explain intent, not mechanics
- early validation with descriptive errors

## Working Rules

### Before Editing

- Read the file you will edit.
- Read at least one nearby reference implementation when style or structure matters.
- Prefer the smallest correct change.

### Tool Use

- Use `Read` for known files.
- Use `Glob` for filename discovery.
- Use `Grep` for content search.
- Parallelize independent reads and searches.
- Use `Explore` only for open-ended codebase questions.

### Editing Style

- Keep methods linear: load -> validate -> compute -> update -> return.
- Extract helpers only for reused logic, dense math, or real domain concepts.
- Keep names concrete and domain-specific.
- Do not add abstraction just to look organized.

### Verification

For functional changes, run the smallest relevant checks first, then broader validation as needed:

```bash
pytest tests/
pytest -k "test_name"
pytest --benchmark-skip
ruff check --fix .
ruff format .
mypy src/pythermondt
pre-commit run --all-files
```

## Common Pitfalls

- Frame operations must keep `Tdata`, `DomainValues`, and `ExcitationSignal` in sync.
- Temporal transforms should preserve or intentionally reset the domain origin.
- Update units when a transform changes physical meaning.
- Prefer tensor operations over loops.
- All functionality changes require tests.

If `AGENTS.md` and local file style disagree, follow the local file style unless it conflicts with an explicit project rule.
