# PyThermoNDT Copilot Instructions

This file is kept because GitHub Copilot uses `.github/copilot-instructions.md` as repository-wide custom instructions. For project architecture, coding style, and domain rules, follow `AGENTS.md` first.

Keep this file intentionally concise.

- Include only non-obvious, high-value guidance.
- Prefer concrete rules and style references over repository overviews.
- Do not duplicate documentation that is easy to discover elsewhere in the repo.

Match the coding style of these reference implementations:

- `src/pythermondt/transforms/sampling.py` - `NonUniformSampling`
- `src/pythermondt/dataset/base_dataset.py` - `BaseDataset`

Default style expectations:

- prefer minimal, linear implementations over extra abstraction
- keep methods easy to scan: load -> validate -> compute -> update -> return
- use helpful Google-style docstrings and sparse comments
- validate early with descriptive errors
- use concrete domain names instead of generic helper names

Project rules:

- preserve temporal consistency across `/Data/Tdata`, `/MetaData/DomainValues`, and `/MetaData/ExcitationSignal`
- update units when physical meaning changes
- prefer tensor operations over Python loops
- all functional changes must include tests
- do not mix functionality and whitespace-only cleanup in the same commit

Useful validation commands:

```bash
pytest tests/
pytest -k "test_name"
pytest --benchmark-skip
ruff check --fix .
ruff format .
mypy src/pythermondt
pre-commit run --all-files
```
