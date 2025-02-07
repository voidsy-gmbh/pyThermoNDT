import io
import h5py
from .base_parser import BaseParser
from ...data import DataContainer

class HDF5Parser(BaseParser):
    @staticmethod
    def parse(data_bytes: io.BytesIO) -> DataContainer:
        """ Parses the data from the given BytesIO object, that was read using one of the BaseReaders subclasses into a DataContainer object.

        The BytesIO object must contain a serialized DataContainer object in HDF5 format.

        Parameters:
            data_bytes (io.BytesIO): The BytesIO object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.

        Raises:
            ValueError: If the given BytesIO object is empty or does not contain a valid HDF5 file.
        """
        # Create a new DataContainer from the BytesIO object and return it
        return DataContainer(data_bytes)