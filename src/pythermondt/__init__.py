import logging

from .__pkginfo__ import __version__
from .config import configure_logging, settings
from .data import DataContainer, ThermoContainer
from .dataset import IndexedThermoDataset, ThermoDataset
from .io import HDF5Parser, SimulationParser
from .readers import AzureBlobReader, LocalReader, S3Reader
from .transforms import augmentation, compose, frequency, normalization, preprocessing, sampling, utils
from .writers import AzureBlobWriter, LocalWriter, S3Writer

# Set up logging per Python best practices: https://docs.python.org/3/howto/logging.html
# Add NullHandler to prevent "No handlers could be found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AzureBlobReader",
    "AzureBlobWriter",
    "DataContainer",
    "HDF5Parser",
    "IndexedThermoDataset",
    "LocalReader",
    "LocalWriter",
    "S3Reader",
    "S3Writer",
    "SimulationParser",
    "ThermoContainer",
    "ThermoDataset",
    "__version__",
    "augmentation",
    "compose",
    "configure_logging",
    "frequency",
    "normalization",
    "preprocessing",
    "sampling",
    "settings",
    "utils",
]

# TODO: Implement async data loading
