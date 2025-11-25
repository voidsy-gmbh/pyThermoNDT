import logging

from .__pkginfo__ import __version__
from .config import configure_logging, settings
from .data import DataContainer, ThermoContainer
from .dataset import IndexedThermoDataset, ThermoDataset
from .io import HDF5Parser, SimulationParser
from .readers import LocalReader, S3Reader
from .transforms import augmentation, normalization, preprocessing, sampling, utils
from .writers import LocalWriter, S3Writer

# Set up logging per Python best practices: https://docs.python.org/3/howto/logging.html
# Add NullHandler to prevent "No handlers could be found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "DataContainer",
    "ThermoContainer",
    "ThermoDataset",
    "IndexedThermoDataset",
    "HDF5Parser",
    "SimulationParser",
    "LocalReader",
    "S3Reader",
    "augmentation",
    "normalization",
    "preprocessing",
    "sampling",
    "utils",
    "LocalWriter",
    "S3Writer",
    "settings",
    "configure_logging",
]

# TODO: Implement async data loading
