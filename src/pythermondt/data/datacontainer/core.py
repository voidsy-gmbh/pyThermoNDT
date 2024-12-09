from typing import Optional
from io import BytesIO
from .node import RootNode
from .group_ops import GroupOps
from .dataset_ops import DatasetOps
from .attribute_ops import AttributeOps
from .serialization_ops import SerializationOps, DeserializationOps
from .visualization_ops import VisualizationOps

class DataContainer(SerializationOps, DeserializationOps, VisualizationOps, GroupOps, DatasetOps, AttributeOps):
    """
    Manages and serializes data into HDF5 format.

    This class manages data in a hierarchical structure, similar to a HDF5 File. It provides methods to add groups, datasets and attributes to the data structure. 
    The data structure can be serialized to a HDF5 file and deserialized from a HDF5 file using save_to_hdf5 and load_from_hdf5 methods respectively.
    It also provides methods for visualization of the data structure.
    """
    def __init__(self, hdf5_bytes: Optional[BytesIO] = None):
        """ Initializes a DataContainer instance.
        By default, initializes an empty DataContainer. 
        If a serialized HDF5 file is provided, the DataContainer is initialized with the data from the BytesIO object.

        Parameters:
            hdf5_bytes (BytesIO, optional): The serialized HDF5 data as a BytesIO object.
        """
        super().__init__()

        # Add the root node to the data structure
        self.nodes["/"] = RootNode()

        # If provided, initialize from a serialized HDF5 file.
        if hdf5_bytes:
            self.deserialize(hdf5_bytes)

   # Overwrite the __str__ method to provide a string representation of the DataContainer
    def __str__(self):
        returnstring = ""
        for path, node in self.nodes.items():
            returnstring = returnstring + f"{path}: ({node.name}: {node.type})" + "\n"

        return returnstring