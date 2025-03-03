import io

from ...data import DataContainer
from .base_parser import BaseParser


class HDF5Parser(BaseParser):
    supported_extensions = (".h5", ".hdf5")

    @staticmethod
    def parse(data_bytes: io.BytesIO) -> DataContainer:
        """Parses the data from the given BytesIO object into a DataContainer object.

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
