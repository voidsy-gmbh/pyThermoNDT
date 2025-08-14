# PyThermoNDT Agents

Hello agent. You are one of the most talented programmers of your generation.

You are looking forward to putting those talents to use to improve PyThermoNDT.

## Philosophy

PyThermoNDT is a PyTorch-compatible package focused on elegant thermal data processing while maintaining scientific rigor and performance.

Every transform must earn its place in the pipeline. Prefer clarity over complexity. We believe that well-designed, composable components can handle the full spectrum of thermographic NDT workflows.

Never mix functionality changes with whitespace changes. All functionality changes must be tested using the existing test infrastructure under `tests/`.

## Style

Follow the project's Ruff configuration with 120-character line limits. Match the existing PyTorch Module patterns and maintain consistency with the thermal data domain conventions.

## Project Overview
PyThermoNDT is a PyTorch-compatible Python package for thermographic data processing in Non-Destructive Testing (NDT). The architecture follows a modular pipeline pattern: **Readers** → **DataContainers** → **Transforms** → **Datasets** → **PyTorch DataLoaders**.

## Core Architecture

### Data Flow Pipeline
1. **Readers** (`src/pythermondt/readers/`) - Load data from sources (local files, S3)
2. **DataContainers** (`src/pythermondt/data/`) - Hierarchical data structure (HDF5-like)
3. **Transforms** (`src/pythermondt/transforms/`) - Processing pipeline (PyTorch nn.Module based)
4. **Datasets** (`src/pythermondt/dataset/`) - PyTorch Dataset interface

### Key Components

#### DataContainer Structure
All data uses the standardized ThermoContainer hierarchy:
```
/Data/Tdata               # Raw thermal data (H×W×T)
/GroundTruth/DefectMask   # Binary defect masks
/MetaData/LookUpTable     # Temperature conversion
/MetaData/DomainValues    # Time values
/MetaData/ExcitationSignal # Heating pattern
```

#### Transform System
- All transforms inherit from `ThermoTransform` (deterministic) or `RandomThermoTransform` (stochastic)
- Use `__init__` parameters for configuration, store as instance attributes
- Implement `forward(container: DataContainer) -> DataContainer`
- Follow PyTorch Module patterns for `extra_repr()` and parameter handling

#### Reader/Backend Separation
- **Readers** handle file discovery and caching logic
- **Backends** handle I/O operations (LocalBackend, S3Backend)
- **Parsers** convert file formats to DataContainers (HDF5Parser, SimulationParser)

## Development Patterns

### Transform Implementation
```python
class MyTransform(ThermoTransform):
    def __init__(self, param: int):
        super().__init__()
        self.param = param

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract datasets
        tdata, domain_values = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues")

        # Process data
        processed_tdata = self._process(tdata)

        # Update container
        container.update_datasets(("/Data/Tdata", processed_tdata))
        return container
```

### Testing Strategy
- Use fixtures from `tests/conftest.py` for common test data
- Test files follow module structure: `tests/transforms/test_sampling.py`
- Integration tests in `tests/integration/` for end-to-end workflows
- Use `pytest.mark.parametrize` for multiple input scenarios

### Configuration
- Global settings in `pythermondt.config.settings` (uses pydantic-settings)
- Environment variables prefixed with `PYTHERMONDT_`
- Download directory and worker count configurable

## Common Workflows

### Development Setup
```bash
uv venv && uv pip install -e . && uv pip install -r requirements_dev.txt
pre-commit install  # Essential for code quality
```

### Running Tests
```bash
pytest tests/                    # All tests
pytest tests/transforms/         # Transform tests only
pytest -k "test_sampling"        # Specific test pattern
```

### Code Quality
- Pre-commit hooks run Ruff (linting/formatting), type checking, security checks
- Configuration in `pyproject.toml` with strict settings
- Use `ruff check --fix` for manual formatting

## Critical Implementation Details

### Time Domain Handling
- Always check `container.get_unit("/MetaData/DomainValues")["quantity"] == "time"` for temporal operations
- Frame selection must adjust domain values: `domain_values - domain_values[0]`
- Maintain temporal consistency across Tdata, DomainValues, and ExcitationSignal

### Unit Management
- Use `Units` enum from `pythermondt.data.units`
- Transforms must preserve or update units appropriately
- LUT application changes Tdata units from `arbitrary` to `kelvin`

### Error Handling Patterns
- Validate tensor dimensions before operations
- Check frame indices bounds: `idx < 0 or idx >= tdata.shape[-1]`
- Raise `ValueError` with descriptive messages for user errors

### Performance Considerations
- Use tensor operations over loops for frame processing
- Leverage `settings.num_workers` for parallel operations
- Consider memory usage for large thermal sequences (H×W×T tensors)

## Integration Points
- **PyTorch**: Transforms are `nn.Module`, datasets follow `torch.utils.data.Dataset`
- **AWS S3**: S3Reader with boto3, supports download caching
- **HDF5**: Primary data format, hierarchical structure matches DataContainer
- **Jupyter**: Interactive analysis methods like `container.analyse_interactive()`

When implementing new features, follow the existing modular patterns and ensure compatibility with the PyTorch ecosystem.
