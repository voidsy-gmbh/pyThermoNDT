import io
from typing import Tuple, Optional
from ._base_reader import _BaseReader
from ..data import DataContainer
from ..transforms import ThermoTransform

class HDF5Reader(_BaseReader):
    '''
    A class for reading DataContainers that have been serialized to HDF5 files.
    '''
    def __init__(self, source: str, file_extension: str | Tuple[str, ...] = '.hdf5', cache_paths: bool = True, transform: Optional[ThermoTransform] = None):
        # Call the constructor of the BaseLoader and set the file extension
        super().__init__(source, file_extension, cache_paths, transform)

    def _read_data(self, file_path: str) -> DataContainer:
        # Read the HDF5 file as BythesIO object
        with open(file_path, 'rb') as file:
            hdf5_bytes = io.BytesIO(file.read())

        # Initialize DataContainer from the BytesIO object and return it
        return DataContainer(hdf5_bytes)