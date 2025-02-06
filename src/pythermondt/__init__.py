from .data import ThermoContainer, ThermoDataset
from .transforms import augmentation, normalization, utils, preprocessing, sampling
from .readers import S3Reader, LocalReader
from .writers import S3Writer, LocalWriter
from .io import HDF5Parser, SimulationParser
from .__pkginfo__ import __version__

# TODO: Implement logging instead of print statements according to this guide: https://docs.python.org/3/howto/logging.html
# TODO: Implement async data loading
# TODO: Implement multi threading for data loading / writing
# TODO: Add more tests
# TODO: Add glob patterns to S3Reader
