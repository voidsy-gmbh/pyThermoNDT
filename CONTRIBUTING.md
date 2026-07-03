# Contributing to PyThermoNDT

Thank you for your interest in contributing to PyThermoNDT! This document provides guidelines and instructions for contributing to the project.

## Pull Request Process

First of all, any contributions to the project are welcome! If you want to contribute, please follow these steps:

1. Create a feature branch from `main`.
2. Make your changes and update the documentation if necessary.
3. Add or update tests for any new or modified functionality.
4. Ensure your code passes all tests and pre-commit hooks.
5. Submit a pull request with:
   - A clear and concise title that describes the changes (this will later appear in the changelog)
   - An appropriate label (this is important for automated changelog generation, see [release.yml](.github/release.yml))
   - A clear description of:
     - What problem your PR solves
     - The changes you've made
     - Any relevant issue numbers

Your pull request will be reviewed by the maintainers, who may request changes or provide feedback.

## Setting Up Development Environment

There are two recommended ways to set up your development environment:

### Option 1: Using UV / Pip (Recommended)

Using uv/Pip is recommended as it is faster and more lightweight than Conda:

```bash
# Set up the virtual environment
uv venv

# Install the package itself in development mode and dev dependencies
uv pip install -e . --group dev
```

This creates a virtual environment called `venv` with all necessary dependencies and installs PyThermoNDT in editable mode. Changes to the source code will immediately be reflected without having to reinstall the package (see [setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) for more information).

If you don't want to use uv, you can also use pip:

```bash
# Create virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate.bat  # Windows

# Install the package itself in development mode and dev dependencies
pip install -e . --group dev
```

### Option 2: Using Conda

The dependencies for development are specified in [environment.yml](environment.yml).

```bash
# Create conda environment from the environment file
conda env create -f environment.yml
```

This creates a conda environment called `pythermondt-dev` with all necessary dependencies installed in development mode.

### Pre-commit Hooks

This project uses pre-commit hooks to enforce code quality standards. The hooks automatically check and fix common issues like trailing whitespace, end-of-file issues, and run Ruff for code formatting and linting. For this project it is highly recommended to use pre-commit hooks to ensure code quality.

To set up pre-commit hooks:

1. Make sure you have installed the development dependencies as described above.

2. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. The hooks will now run automatically on every commit. If you want to run them manually, use:
   ```bash
   pre-commit # Runs only on staged files
   pre-commit run --all-files # Runs on all files
   ```

Pre-commit configuration is defined in `.pre-commit-config.yaml` and includes:
- Basic formatting fixes (trailing whitespace, end-of-file fixer)
- YAML and TOML syntax checking
- Large file detection
- Security checks (detect private keys)
- Ruff for code linting and formatting

After you have set up the environment and installed the pre-commit hooks, you can start developing.

If you are using VSCode, make sure to select the correct interpreter for the environment you have created to enable proper autocompletion.
For more information see the [VSCode documentation](https://code.visualstudio.com/docs/python/environments).

## Code Quality and Validation

This project uses automated testing, linting, and code formatting to maintain code quality.

### Running Tests

Tests are written using pytest and are located in the [tests](tests/) directory.

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_file.py::test_name

# Filter tests by name
pytest -k test_name
```

### Performance Benchmarking

PyThermoNDT includes a benchmarking framework for measuring transform performance using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).

```bash
# Run all benchmarks
pytest tests/benchmark/ --benchmark-only

# Run only local benchmarks (no cloud data access)
pytest tests/benchmark/ --benchmark-only -m "local"

# Benchmark specific transforms
pytest tests/benchmark/ --benchmark-only -k "ApplyLUT"
```

Use the `-m "local"` marker to run benchmarks without requiring AWS credentials.

###  Code Formatting and Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting to maintain consistent code quality. Ruff combines functionality from multiple tools (black, flake8, isort, pylint, etc.) into a single, fast package written in Rust.

#### Running Ruff

```bash
# Format code
ruff format

# Check for any linting issues
ruff check

# Check and fix linting issues where possible
ruff check --fix
```

Optionally, you can install the [VSCode extension for Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for real-time feedback.

## Versioning

The package version is defined statically in `pyproject.toml` (`[project].version`) following [PEP 440](https://peps.python.org/pep-0440/) using the canonical `.dev0` form for developmental releases. The `__version__` attribute on the `pythermondt` package is populated at runtime via `importlib.metadata.version("pythermondt")`.

### Release Workflow

This project uses an automated release workflow with GitHub Actions and [`uv version`](https://docs.astral.sh/uv/reference/cli/#uv-version). Manually updating the version string should be avoided. If you really need to, use `uv version --bump stable` for release stabilization, or derive an explicit development version with `uv version --bump <component> --dry-run --short` and append `.dev0`.

#### Release Branch Convention

- The `main` branch always carries a `.dev0` suffix (e.g. `0.3.7.dev0`).
- Release branches carry the stable version (e.g. `0.3.7`).
- Git tags are created **manually** on the merge commit when publishing a release — the automated workflows never create tags.

#### Steps to Release

1. **Prepare Release Branch:**
   - Manually trigger the [Prepare Release workflow](https://github.com/voidsy-gmbh/pyThermoNDT/actions/workflows/prepare_release.yml) on the `main` branch
   - This workflow:
     - Creates a `release/X.Y.Z` branch
     - Bumps the version by removing the `.dev0` suffix (`uv version --bump stable`)
     - Commits the bump and pushes the release branch

2. **Create and Merge Release PR:**
   - Manually create a pull request from `release/X.Y.Z` to `main`
   - Wait for automated checks (tests and code quality) to pass
   - Review and merge the PR

3. **Publish Release:**
   - Go to GitHub's [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases)
   - Click "Draft a new release"
   - Create a new tag: `X.Y.Z` on the merge commit on `main`
   - Set title: `vX.Y.Z`
   - Generate release notes
   - Click "Publish release"

4. **Automatic Post-Release:**
   - Publishing triggers the [Post-Release workflow](https://github.com/voidsy-gmbh/pyThermoNDT/actions/workflows/post_release.yml) which:
     - Builds the package from the release tag
     - Publishes to PyPI
     - Signs and uploads artifacts to the GitHub release
     - Bumps `main` to the next patch version with a `.dev0` suffix (e.g. `0.3.8.dev0`)
