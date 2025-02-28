# pyThermoNDT

pyThermoNDT is a Python package for manipulating thermographic data in Non-Destructive Testing (NDT) applications. It provides various methods to load, transform, visualize, and write thermographic data, making it easier and more efficient to work with thermal imaging in NDT contexts.

## Installation

### From Wheel Package (Recommended)
Download the latest wheel package (`.whl` file) from the [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases) and run:

```bash
pip install /path/to/downloaded/pythermondt-x.y.z-py3-none-any.whl
```

### From Source
pyThermoNDT can be installed directly from the source code:

Clone the repository and install the package locally:

```bash
git clone https://github.com/voidsy-gmbh/pyThermoNDT.git
cd pyThermoNDT
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/voidsy-gmbh/pyThermoNDT.git
```

### From a local conda channel
Download the latest conda install packge (`.zip` file) from the [releases page](github.com/voidsy-gmbh/pyThermoNDT/releases) and unzip it. The file contains a local conda channel
with pyThermoNDT and its dependencies. To install pyThermoNDT from the local channel, run:

```bash
conda config --add channels conda-forge # Recommended to resolve pip dependencies using conda
conda install pythermondt -c /path/to/downloaded/folder
```

## Documentation
For detailed usage examples, check out the Jupyter Notebooks in the [examples](examples/) directory.

## Contributing
Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details on [setting up a development environment](CONTRIBUTING.md#setting-up-development-environment), [coding standards](CONTRIBUTING.md#code-quality-and-validation), and the [pull request process](CONTRIBUTING.md#pull-request-process).
