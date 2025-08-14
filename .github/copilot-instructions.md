# PyThermoNDT AI Coding Instructions

PyThermoNDT is a PyTorch-compatible package for thermographic data processing in Non-Destructive Testing (NDT). This guide helps AI agents understand the essential architecture and patterns for immediate productivity.

## Core Architecture Pattern

**Data Flow:** Readers → DataContainers → Transforms → Datasets → PyTorch DataLoaders

1. **Readers** (`src/pythermondt/readers/`) coordinate Backends (I/O) + Parsers (format conversion)
2. **DataContainers** (`src/pythermondt/data/datacontainer/`) provide hierarchical HDF5-like data structure
3. **Transforms** (`src/pythermondt/transforms/`) are PyTorch nn.Modules for processing pipelines
4. **Datasets** (`src/pythermondt/dataset/`) bridge to PyTorch DataLoader interface

## Essential DataContainer Structure

All thermal data follows standardized ThermoContainer hierarchy:
```
/Data/Tdata               # Raw thermal data (H×W×T tensors)
/GroundTruth/DefectMask   # Binary defect masks
/MetaData/LookUpTable     # Temperature conversion
/MetaData/DomainValues    # Time values for frames
/MetaData/ExcitationSignal # Heating pattern
```

**Key API Pattern:**
```python
# Extract datasets
tdata, domain_values = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues")

# Process data
processed_tdata = self._process(tdata)

# Update container (preserves hierarchy)
container.update_datasets(("/Data/Tdata", processed_tdata))
```

## Transform Implementation Pattern

All transforms inherit from `ThermoTransform` (deterministic) or `RandomThermoTransform` (stochastic):

```python
class MyTransform(ThermoTransform):
    def __init__(self, param: int):
        super().__init__()
        self.param = param  # Store as instance attribute

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract, process, update pattern
        tdata = container.get_dataset("/Data/Tdata")
        processed = self._process(tdata)
        container.update_datasets(("/Data/Tdata", processed))
        return container
```

**Critical Implementation Details:**
- Use `container.get_datasets()` for multiple datasets in one call
- Check time units: `container.get_unit("/MetaData/DomainValues")["quantity"] == "time"`
- Validate tensor bounds: `idx < 0 or idx >= tdata.shape[-1]` raises `ValueError`
- Frame operations must adjust domain values: `domain_values - domain_values[0]`

## Development Workflow

**Setup:**
```bash
uv venv && uv pip install -e . && uv pip install -r requirements_dev.txt
pre-commit install  # Essential - runs Ruff, mypy, typos
```

**Testing Pattern:**
```bash
pytest tests/transforms/test_sampling.py  # Specific transform tests
pytest -k "test_apply_lut"                # Pattern matching
pytest tests/integration/                 # End-to-end workflows
```

**Code Quality:**
- Ruff enforces 120-char lines, Google docstring style
- Configuration in `pyproject.toml` with strict linting
- Pre-commit hooks block commits with quality issues
- Use `pytest.mark.parametrize` for multiple test scenarios

## Project-Specific Conventions

**Configuration:**
- Global settings via `pythermondt.config.settings` (pydantic-settings)
- Environment variables prefixed `PYTHERMONDT_` (e.g., `PYTHERMONDT_DOWNLOAD_DIR`)
- Settings auto-create missing directories and validate worker counts

**Unit Management:**
- Use `Units` enum from `pythermondt.data.units`
- LUT application changes Tdata units from `arbitrary` to `kelvin`
- Transforms must preserve/update units appropriately

**Error Patterns:**
- `ValueError` for user input errors with descriptive messages
- `IndexError` for frame index bounds violations
- Validate tensor dimensions before operations

**Performance Considerations:**
- Use tensor operations over loops for frame processing
- Leverage `settings.num_workers` for parallel operations
- Consider memory usage for large H×W×T thermal sequences

## File Organization Patterns

**Test Structure:** Mirror source structure
- `tests/transforms/test_sampling.py` tests `src/pythermondt/transforms/sampling.py`
- Use fixtures from `tests/conftest.py` for common test data
- Integration tests in `tests/integration/` for full workflows

**Transform Categories:**
- `preprocessing.py` - ApplyLUT, RemoveFlash, SubtractFrame
- `sampling.py` - NonUniformSampling, SelectFrames, SelectFrameRange
- `normalization.py` - MinMaxNormalize, ZScoreNormalize, MaxNormalize
- `augmentation.py` - GaussianNoise, RandomFlip (stochastic transforms)

## Critical Integration Points

**PyTorch Compatibility:**
- Transforms are `nn.Module` subclasses with proper `forward()` methods
- Datasets implement `torch.utils.data.Dataset` interface
- Use `container_collate()` function for DataLoader batching

**AWS S3 Integration:**
- S3Reader with boto3, supports download caching to `settings.download_dir`
- Automatic backend selection (LocalBackend vs S3Backend)

**Time Domain Handling:**
- Always verify time units before temporal operations
- Frame selection must maintain temporal consistency across Tdata, DomainValues, ExcitationSignal
- Use `domain_values[selected_indices]` for frame subsets

When implementing features, follow the modular Reader/Transform/Dataset pattern and ensure PyTorch ecosystem compatibility.
