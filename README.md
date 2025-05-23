# pyThermoNDT

pyThermoNDT is a Python package for manipulating thermographic data in Non-Destructive Testing (NDT) applications. It provides various methods to load, transform, visualize, and write thermographic data, making it easier and more efficient to work with thermal imaging in NDT contexts.

## Features

- **Multi-source Data Loading:** Read thermographic data seamlessly from local files and S3 storage
- **Hierarchical Data Structure:** Store and access thermographic data, metadata, and ground truth in a common format
- **Remote Data Caching:** Optionally cache data from remote sources for improved performance
- **Composable Transforms:** Build custom processing pipelines with reusable transform components
- **PyTorch Integration:** Datasets compatible with PyTorch DataLoader for training deep learning models

## Quick Example
```python
from torch.utils.data import DataLoader

from pythermondt import transforms as T
from pythermondt.data import ThermoDataset, container_collate
from pythermondt.readers import LocalReader, S3Reader

# Load data from different sources
local_reader = LocalReader("path/to/data/*.hdf5", cache_files=True)
s3_reader = S3Reader("s3://ffg-bp/example4/.hdf5", cache_files=True)

# Create a transform pipeline
transform = T.Compose([
    T.ApplyLUT(),           # Convert raw data to temperatures
    T.RemoveFlash(),        # Remove flash frames
    T.NonUniformSampling(64), # Resample data to 64 frames
    T.CropFrames(96, 96), # Center crop the frames to 96x96
    T.MinMaxNormalize()     # Normalize data
])

# 1.) Access data using the reader interface
container = local_reader[0]
processed = transform(container) # Apply the transform to the container

# Visualize results
processed.show_frame(frame_number=10)

# 2.) Combine multiple sources in a dataset with caching for remote data and applied transforms
dataset = ThermoDataset([local_reader, s3_reader], transform=transform)

# Analyse a datacontainer interactively
dataset[0].analyse_interactive()

# 3.) Use the dataset with a PyTorch DataLoader for batched access
# Create a custom collate function to extract thermal data and ground truth
collate_fn = container_collate('/Data/Tdata', '/GroundTruth/DefectMask')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Run your training loop
for thermal_data, ground_truth in dataloader:
    print(f"Thermal data shape: {thermal_data.shape}")    # Tensor of shape: [32, 64, 96, 96]
    print(f"Ground truth shape: {ground_truth.shape}")    # Tensor of shape: [32, 96, 96]
    break
```

## From here?
pyThermoNDT is yours to use! You can start by exploring the [examples](examples/) directory for more detailed usage scenarios. The package is designed to be flexible and extensible, so feel free to modify and adapt it to your specific needs.

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
