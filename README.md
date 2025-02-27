# pyThermoNDT

pyThermoNDT is a Python package for manipulating thermographic data in Non-Destructive Testing (NDT) applications. It provides various methods to load, transform, visualize, and write thermographic data, making it easier and more efficient to work with thermal imaging in NDT contexts.

## Installation

### Pip

#### From Wheel Package (Recommended)
1. Download the wheel package (`.whl` file) from the [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases)
2. Install using pip:
    ```
    pip install /path/to/downloaded/file.whl
    ```

#### From Source
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/pyThermoNDT.git
    cd pyThermoNDT
    ```
2. Install either in development mode:
    ```
    pip install -e .
    ```
    or just install the package:
    ```
    pip install .
    ```

### Conda

#### From Install Package (Recommended)
The current recommended way to install pythermondt via conda is to use the install package, provided as a .zip files with the releases:

1. Download the latest conda install package (`.zip` file) from the [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases).

2. Add conda-forge to your channels. This is recommended so that pip dependencies can be resolved without having to specifiy the channel each time:
    ```
    conda config --add channels conda-forge
    ```

3. Unpack the .zip file. Now install pyThermoNDT in your current environment using conda:
    ```
    conda install pythermondt -c /path/to/unpacked/folder
    ```

#### From Source
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/pyThermoNDT.git
    cd pyThermoNDT
    ```

2. Add conda-forge to your channels. This is recommended so that pip dependencies can be resolved without having to specifiy the channel each time:
    ```
    conda config --add channels conda-forge
    ```

3. Build the package (Note: you need to install the [conda-build](https://docs.conda.io/projects/conda-build/en/stable/install-conda-build.html) package for this):
   ```
    conda build conda
   ```

4. Install the built package in your current environment:
   ```
   conda install pythermondt --use-local
   ```

## Documentation
For detailed usage examples, check out the Jupyter Notebooks in the [examples](examples/) directory.

## Contributing
Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details on setting up a development environment, coding standards, and pull request process.
