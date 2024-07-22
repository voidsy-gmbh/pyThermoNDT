import h5py
import numpy as np
from typing import Tuple
from ._base_reader import _BaseReader
from ..data import DataContainer

class HDF5Reader(_BaseReader):
    '''
    A class for reading DataContainers that have been serialized to HDF5 files.
    '''
    def __init__(self, source: str, file_extension: str | Tuple[str, ...] = '.hdf5', cache_paths: bool = True):
        # Call the constructor of the BaseLoader and set the file extension
        super().__init__(source, file_extension, cache_paths)

    def _read_data(self, file_path: str) -> DataContainer:
        # Create an empty DataContainer
        datacontainer = DataContainer()

        # Read data from the specified HDF5 file
        with h5py.File(file_path, 'r') as f:
            
            # Iterate over all groups in the file
            for group_name , group in f.items():

                # If any of the groups has attributes, add them to the group in the DataContainer
                if group.attrs:
                    datacontainer.update_attributes(group_name, **group.attrs)

                # Iterate over all datasets in the group
                for dataset_name, dataset in group.items():
                    # Add dataset attributes if they exist
                    if dataset.attrs:
                        datacontainer.update_attributes("/".join([group_name, dataset_name]), **dataset.attrs)

                    # Fill the dataset in the DataContainer with the data from the HDF5 file
                    datacontainer.fill_dataset("/".join([group_name, dataset_name]), dataset[()])        

        # Return the filled datacontainer
        return datacontainer