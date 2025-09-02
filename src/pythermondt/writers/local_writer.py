import os

from ..data import DataContainer
from ..io import IOPathWrapper, LocalBackend
from .base_writer import BaseWriter


class LocalWriter(BaseWriter):
    def __init__(self, destination_folder: str):
        """Instantiates a new HDF5Writer.

        Args:
            destination_folder (str): The destination folder where the DataContainers should be written to.
        """
        super().__init__()

        # Verify folder
        self.__exists = os.path.exists(destination_folder)
        self.__destination_folder = destination_folder

    def _create_backend(self) -> LocalBackend:
        return LocalBackend(pattern=self.__destination_folder)

    def write(self, container: DataContainer, file_name: str):
        # Verify folder
        if self.__exists or not os.path.exists(self.__destination_folder):
            os.makedirs(self.__destination_folder, exist_ok=True)

        # Append file extension if not present
        if not file_name.endswith(".hdf5"):
            file_name += ".hdf5"

        # Create the path to the file
        path = os.path.join(self.__destination_folder, file_name)

        # Write the DataContainer to the file
        self.backend.write_file(IOPathWrapper(container.serialize_to_hdf5()), path)
