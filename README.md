# pyThermoNDT

pyThermoNDT is a Python package for manipulating thermographic data in Non-Destructive Testing (NDT) applications. It provides various methods to load, transform, visualize, and write thermographic data, making it easier and more efficient to work with thermal imaging in NDT contexts.

## Installation

### From Release Package (Recommended)
The current recommended way to install pythermondt is to use the .zip files provided with the releases:

1. Download the latest release package from the [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases) and unpack it.

2. Add conda-forge to your channels. This is recommended so that pip dependencies can be resolved without having to specifiy the channel each time:
    ```
    conda config --add channels conda-forge
    ```

3. Install the package in your current environment using conda:
    ```
    conda install pythermondt -c /path/to/unpacked/folder
    ```

### From Source
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

## Usage
For using this package, take a look at the Jupyter Notebooks in [examples](examples/).

## Development

### Setting up the development environment

The dependencies for development are specified in [environment.yml](environment.yml)

To set up the environment run the following command in the root directory of the repository:

```
conda env create
conda activate pythermondt-dev
```

### Running Tests
Tests are written using pytest and are located in [tests](tests/). Tests can be run locally using the following command:

```
pytest
```