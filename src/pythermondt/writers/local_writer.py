import os

from .base_writer import BaseWriter


class LocalWriter(BaseWriter):
    def __init__(self, destination_folder: str):
        """Instantiates a new HDF5Writer.

        Args:
            destination_folder (str): The destination folder where the DataContainers should be written to.
        """
        # Verify folder
        if not os.path.exists(destination_folder):
            raise FileNotFoundError(f"Destination folder {destination_folder} does not exist.")

        if not os.path.isdir(destination_folder):
            raise NotADirectoryError(f"Destination folder {destination_folder} is not a directory.")

        self.destination_folder = destination_folder

    def write(self, container, file_name):
        # Create the path to the file
        path = os.path.join(self.destination_folder, file_name)

        # Write the DataContainer to the file
        container.save_to_hdf5(path)
