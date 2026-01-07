# PyThermoNDT Agent Instructions

Hello agent. You are one of the most talented programmers of your generation, ready to improve PyThermoNDT - a PyTorch-compatible package for thermographic Non-Destructive Testing (NDT) data processing.

## Philosophy

- **Clarity over Complexity**: Every transform must earn its place in the pipeline
- **Scientific Rigor**: Maintain physical accuracy in thermal data processing
- **PyTorch Compatible**: Seamless integration with PyTorch workflows
- **Never mix functionality and whitespace changes** - separate commits
- **All functionality changes must include tests**

## Architecture

```
Readers → DataContainers → Transforms → Datasets → PyTorch DataLoaders
```

**Tech Stack**: Python 3.10-3.14, PyTorch ≥2.0, NumPy, h5py, boto3 (S3), pytest, ruff, mypy

### DataContainer Structure (Standard Paths)

```
/Data/Tdata                    # Thermal data (H×W×T)
/GroundTruth/DefectMask        # Defect masks (H×W)
/MetaData/LookUpTable          # Temperature conversion (Uint16→Float64)
/MetaData/DomainValues         # Time values (T,)
/MetaData/ExcitationSignal     # Heating pattern
```

**Access Pattern**:
```python
# Get datasets (efficient for multiple)
tdata, mask = container.get_datasets("/Data/Tdata", "/GroundTruth/DefectMask")

# Update datasets
container.update_datasets(
    ("/Data/Tdata", processed_tdata),
    ("/MetaData/DomainValues", new_domain_values)
)
```

### Transform Pattern

```python
class MyTransform(ThermoTransform):  # or RandomThermoTransform for stochastic
    def __init__(self, param: int):
        super().__init__()
        self.param = param

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")

        # Validate early
        if tdata.ndim != 3:
            raise ValueError(f"Expected 3D tensor (H×W×T), got {tdata.shape}")

        # Process (use tensor ops, not loops)
        processed = self._process(tdata)

        container.update_dataset("/Data/Tdata", processed)
        return container

    def extra_repr(self) -> str:
        return f"param={self.param}"
```

## Code Conventions

**Ruff** (line length: 120, Google docstrings, double quotes):
```python
def method(self, param: int, optional: str | None = None) -> DataContainer:
    """Short summary ending with period.

    Args:
        param (int): Description.
        optional (str, optional): Description. Defaults to None.

    Returns:
        DataContainer: Description.

    Raises:
        ValueError: When param is invalid.
    """
```

**Type Hints** (modern Python 3.10+):
```python
def process(data: list[int]) -> dict[str, float] | None:  # Not List, Optional
    pass
```

**Naming**:
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Private: `__double_underscore`

**Errors** (descriptive with context):
```python
raise ValueError(f"Frame {idx} out of range [0, {total_frames})")
```

## Testing

**Test organization**:
- Tests organized by domain: `tests/{data,dataset,io,integration}/test_*.py`
- Use fixtures from `tests/conftest.py`

**Pattern**:
```python
@pytest.mark.parametrize("num_frames,expected", [(10, (96,96,10)), (32, (96,96,32))])
def test_feature(sample_container, num_frames, expected):
    # Test implementation
    assert result.shape == expected
```

**Common commands**:
```bash
pytest tests/                    # All tests
pytest -k "test_name"           # Pattern match
pytest --benchmark-skip         # Skip benchmarks (faster)
ruff check --fix .              # Lint and fix
mypy src/pythermondt            # Type check
pre-commit run --all-files      # All quality checks
```

## Critical Details

### Temporal Consistency
When selecting frames, **always update temporal metadata together**:
```python
# Select frames
new_tdata = tdata[..., indices]
new_domain_values = domain_values[indices] - domain_values[indices[0]]  # Zero-base!

# Update together
container.update_datasets(
    ("/Data/Tdata", new_tdata),
    ("/MetaData/DomainValues", new_domain_values)
)
```

### Unit Management
```python
from pythermondt.data.units import Units

# Update units when transformation changes physical meaning
container.set_unit("/Data/Tdata", Units.KELVIN)  # After ApplyLUT: arbitrary→kelvin
```

### Performance
- Use tensor operations over loops: `processed = tdata * scale` not `for i in range(...)`
- Validate dimensions early: `if tdata.ndim != 3: raise ValueError(...)`
- Leverage `settings.num_workers` for parallelism

## Key Locations

**Transforms**: `src/pythermondt/transforms/{base,preprocessing,sampling,normalization,augmentation}.py`
**Readers**: `src/pythermondt/readers/{local_reader,s3_reader}.py`
**Tests**: `tests/conftest.py` (fixtures), `tests/{data,dataset,io,integration}/test_*.py`
**Config**: `pyproject.toml`, `src/pythermondt/config.py`

## Development Workflow

```bash
# Setup
uv venv && uv pip install -e . && uv pip install -r requirements_dev.txt && pre-commit install

# Quality checks (run before commit)
ruff check --fix . && ruff format . && mypy src/pythermondt && pytest tests/
```

## Essential Rules

1. Never mix functionality and whitespace changes
2. All functionality changes must include tests
3. Run pre-commit hooks before committing
4. Validate tensor dimensions early with clear errors
5. Maintain temporal consistency (Tdata, DomainValues, ExcitationSignal)
6. Update units when physical meaning changes
7. Use tensor operations over loops
8. Follow PyTorch Module patterns (inherit from ThermoTransform/RandomThermoTransform)

Your contributions should feel native to the codebase. Follow patterns, maintain consistency, prioritize clarity and correctness.

Welcome aboard!
