# PyThermoNDT

PyThermoNDT is a Python package for manipulating thermographic data in Non-Destructive Testing (NDT) applications. It provides various methods to load, transform, visualize, and write thermographic data, making it easier and more efficient to work with thermal imaging in NDT contexts.

## Features

- **Multi-source Data Loading:** Read thermographic data seamlessly from local files and S3 storage
- **Hierarchical Data Structure:** Store and access thermographic data, metadata, and ground truth in a common format
- **Remote Data Caching:** Optionally cache data from remote sources for improved performance
- **Composable Transforms:** Build custom processing pipelines with reusable transform components
- **PyTorch Integration:** Datasets compatible with PyTorch DataLoader for training deep learning models

## Quick Example
```python
from torch.utils.data import DataLoader

from pythermondt import LocalReader, S3Reader
from pythermondt import transforms as T
from pythermondt.dataset import ThermoDataset, container_collate

# Load data from different sources
local_reader = LocalReader("./examples/example_data/**/*.hdf5", recursive=True)
s3_reader = S3Reader("ffg-bp", "example2_writing_data", download_files=True)

# Create optimized transform pipeline (deterministic transforms first for better caching)
transform = T.Compose([
    T.ApplyLUT(),                  # Convert raw data to temperatures
    T.RemoveFlash(),               # Remove flash frames
    T.NonUniformSampling(64),      # Resample data to 64 frames
    T.CropFrames(96, 96),          # Center crop the frames to 96x96
    T.MinMaxNormalize()            # Normalize data
])

# 1.) Access individual files using readers
container = local_reader[0]
processed = transform(container)

# 2.) Analyse processed data
processed.show_frame(frame_number=10)
processed.analyse_interactive()

# 3.) Combine sources in a dataset for training workflows
dataset = ThermoDataset([local_reader, s3_reader], transform=transform)

# 4.) Build cache for faster training (splits pipeline at first random transform)
dataset.build_cache("immediate")

# 5.) Use with PyTorch DataLoader for model training to be used in your training loop
collate_fn = container_collate('/Data/Tdata', '/GroundTruth/DefectMask')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

for epoch in range(50):
    print(f"Epoch {epoch + 1}")
    for thermal_data, ground_truth in dataloader:
        print(f"Thermal data shape: {thermal_data.shape}")    # [4, 96, 96, 64]
        print(f"Ground truth shape: {ground_truth.shape}")    # [4, 96, 96]
```

## From here?
PyThermoNDT is yours to use! You can start by exploring the [examples](examples/) directory for more detailed usage scenarios. The package is designed to be flexible and extensible, so feel free to modify and adapt it to your specific needs.

## Installation

### From Wheel Package (Recommended)
Download the latest wheel package (`.whl` file) from the [release page](https://github.com/voidsy-gmbh/pyThermoNDT/releases) and run:

```bash
pip install /path/to/downloaded/pythermondt-x.y.z-py3-none-any.whl
```

### From Source
PyThermoNDT can be installed directly from the source code:

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
Download the latest conda install package (`.zip` file) from the [release page](https://github.com/voidsy-gmbh/pyThermoNDT/releases) and unzip it. The file contains a local conda channel
with PyThermoNDT and its dependencies. To install PyThermoNDT from the local channel, run:

```bash
conda config --add channels conda-forge # Recommended to resolve pip dependencies using conda
conda install pythermondt -c /path/to/downloaded/folder
```

## Documentation
For detailed usage examples, check out the Jupyter Notebooks in the [examples](examples/) directory.

## Contributing
Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details on [setting up a development environment](CONTRIBUTING.md#setting-up-development-environment), [coding standards](CONTRIBUTING.md#code-quality-and-validation), and the [pull request process](CONTRIBUTING.md#pull-request-process).
