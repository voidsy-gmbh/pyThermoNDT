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

## Usage
For using this package, take a look at the Jupyter Notebooks in [examples](examples/).

## Contributing

### Setting up the Development Environment

You can set up your development environment using either **Conda** or **UV / Pip**.

#### Option 1: Using UV / Pip (Recommended)
Setting up the development environment using UV/Pip is recommended as it is faster and more lightweight than Conda:

To set up the venv run the following command in the root directory of the repository:
```
uv venv
uv pip install -e .[dev]
```
This will create a new virtual environment called `venv` with all the necessary dependencies installed. It will also install pyThermoNDT in development mode. This means that changes to the source code will immediately be reflected in the environment without having to reinstall the package every time. For more information on development mode see the [setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/development_mode.html). Use Select Interpreter in VSCode to select the venv as the interpreter for the workspace. This will allow VSCode to use the venv for running the code and provide proper code completion.

If you dont want to use uv, you can also do the same thing with pip (though pip is much slower than uv):
```
python -m venv venv
source venv/bin/activate # Linux
venv\Scripts\activate.bat # Windows
pip install -e .[dev]
```

#### Option 2: Using Conda
The dependencies for development are specified in [environment.yml](environment.yml)

To set up the environment run the following command in the root directory of the repository:
```
conda env create -f environment.yml
```
This will create a new conda environment called `pythermondt-dev` with all the necessary dependencies installed. It will also install pyThermoNDT in development mode. This means that changes to the source code will immediately be reflected in the environment without having to reinstall the package every time. For more information on development mode see the [setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/development_mode.html). Use Select Interpreter in VSCode to select the conda environment as the interpreter for the workspace. This will allow VSCode to use the conda environment for running the code and provide proper code completion.

### Running Tests
Tests are written using pytest and are located in [tests](tests/). Tests can be run locally using the following command:
```
pytest
```
Single tests can be run by specifying the test file and test name:
```
pytest tests/test_file.py::test_name
```

### Versioning
The package version is defined in [pythermondt/\__pkginfo\__.py](src/pythermondt/__pkginfo__.py). The version number should be updated according to the [PEP 440 – Version Identification and Dependency Specification](https://setuptools.pypa.io/en/latest/userguide/distribution.html) guidelines.

When releasing a new version:
1. Update the version number in [pythermondt/\__pkginfo\__.py](src/pythermondt/__pkginfo__.py).
2. Create a new release branch named 'release/version_number' This will automatically trigger a GitHub action that will build the package and create a release draft on GitHubs [releases page](https://github.com/voidsy-gmbh/pyThermoNDT/releases).