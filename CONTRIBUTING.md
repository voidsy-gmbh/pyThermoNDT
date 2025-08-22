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
uv pip install -e .
uv pip install -r requirements_dev.txt
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
pip install -e .
pip install -r requirements_dev.txt
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

The package version is defined in [pythermondt/\__pkginfo\__.py](src/pythermondt/__pkginfo__.py). The version number should follow [PEP 440](https://setuptools.pypa.io/en/latest/userguide/distribution.html) guidelines.

### Release Workflow

This project uses an automated release workflow with GitHub Actions und [bump-my-version](https://github.com/callowayproject/bump-my-version) Manually updating the version
string should be avoided. If you really need to use the [CLI-Interface](https://callowayproject.github.io/bump-my-version/reference/cli/) that bump-my-version provides.

When using the automated release workflow, the these steps should be followed:
1. **Trigger a release:**
   - Manually trigger the [Build and Release Python Package workflow](https://github.com/voidsy-gmbh/pyThermoNDT/actions/workflows/build_and_release.yml) on the `main` branch.
   - This workflow:
     - Creates a `release/X.Y.Z` branch
     - Bumps the version number by removing the "dev" suffix
     - Builds and tests the package
     - Creates a draft release on GitHub's [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases)

2. **Finalizing a Release**:
   - Review the draft release on GitHub
   - Edit release notes as needed
   - Click "Publish release" to make it official
   - Publishing automatically triggers a post-release workflow that:
     - Increments the patch version in the main branch
     - Adds the "dev" suffix for the next development cycle
