import torch
from typing import Optional
from io import BytesIO
from .node import RootNode, GroupNode, DataNode, NodeType
from .group_ops import GroupOps
from .dataset_ops import DatasetOps
from .attribute_ops import AttributeOps
from .serialization_ops import SerializationOps, DeserializationOps
from .visualization_ops import VisualizationOps
from .utils import is_rootnode, is_groupnode, is_datanode

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
    
    # Overwrite the __eq__ method to provide a comparison between two DataContainer instances
    def __eq__(self, other: object) -> bool:
        """Compare two DataContainers for equality.
    
        Containers are equal if they have:
        1. Same node structure 
        2. Equal data in all datasets
        3. Identical attributes for every group and dataset

        **Note**: This implements strict equality. Even with the same initial data, containers 
        that have undergone stochastic transforms (e.g. GaussianNoise) will not be equal 
        since their data differs.

        Parameters:
            other (object): The other object to compare with.

        Returns:
            bool: True if the two DataContainers are equal, False otherwise.
        """
        # Check if the other object is an instance of DataContainer
        if not isinstance(other, DataContainer):
            return False

        # Check if node structure is equal
        if set(self.nodes.keys()) != set(other.nodes.keys()):
            return False
        
        # Compare each node
        for path, node in self.nodes.items():
            # Retrieve other node
            other_node = other.nodes[path]

            # Compare node types and names
            if node.type != other_node.type or node.name != other_node.name:
                return False
            
            # Checks for GroupNodes
            if is_groupnode(node) and is_groupnode(other_node):
                # Compare attributes
                if dict(node.attributes) != dict(other_node.attributes):
                    return False
            
            # Checks for DataNodes
            if is_datanode(node) and is_datanode(other_node):
                # Compare attributes
                if dict(node.attributes) != dict(other_node.attributes):
                    return False
                    
                # Compare actual data
                if not torch.equal(node.data, other_node.data):
                    return False
        return True