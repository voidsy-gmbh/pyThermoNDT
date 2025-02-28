# pyThermoNDT

pyThermoNDT is a Python package for manipulating thermographic data in Non-Destructive Testing (NDT) applications. It provides various methods to load, transform, visualize, and write thermographic data, making it easier and more efficient to work with thermal imaging in NDT contexts.

## Features

- **Multi-source Data Loading:** Read thermographic data seamlessly from local files and S3 storage
- **Hierarchical Data Structure:** Store and access thermographic data, metadata, and ground truth in a consistent format
- **Remote Data Caching:** Automatically cache data from remote sources for improved performance
- **Composable Transforms:** Build custom processing pipelines with reusable transformation components
- **PyTorch Integration:** Use with PyTorch DataLoader for efficient batch processing and model training

## Quick Example
```python
from pythermondt import transforms as T
from pythermondt.data import ThermoDataset
from pythermondt.readers import LocalReader, S3Reader

# Load data from different sources
local_reader = LocalReader("data/*.hdf5")
s3_reader = S3Reader("s3://bucket-name/data.hdf5")

# Combine into a dataset (with caching for remote data)
dataset = ThermoDataset([local_reader, s3_reader])

# Create a transform pipeline
transform = T.Compose([
    T.ApplyLUT(),           # Convert raw data to temperatures
    T.RemoveFlash(),        # Remove flash frames
    T.NonUniformSampling(64), # Resample data to 64 frames
    T.MinMaxNormalize()     # Normalize data
])

# Access and process first container
container = dataset[0]
processed = transform(container)

# Visualize results
processed.show_frame(frame_number=10)
```

## Installation

### From Wheel Package (Recommended)
Download the latest wheel package (`.whl` file) from the [release page](https://github.com/voidsy-gmbh/pyThermoNDT/releases) and run:

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
Download the latest conda install packge (`.zip` file) from the [release page](https://github.com/voidsy-gmbh/pyThermoNDT/releases) and unzip it. The file contains a local conda channel
with pyThermoNDT and its dependencies. To install pyThermoNDT from the local channel, run:

```bash
conda config --add channels conda-forge # Recommended to resolve pip dependencies using conda
conda install pythermondt -c /path/to/downloaded/folder
```

## Documentation
For detailed usage examples, check out the Jupyter Notebooks in the [examples](examples/) directory.

## Contributing
Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details on [setting up a development environment](CONTRIBUTING.md#setting-up-development-environment), [coding standards](CONTRIBUTING.md#code-quality-and-validation), and the [pull request process](CONTRIBUTING.md#pull-request-process).
