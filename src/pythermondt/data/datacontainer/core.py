from typing import Dict, Any
from functools import wraps
from torch import Tensor
from numpy import ndarray
from .node import Node, RootNode
from .dataset_ops import DatasetOps
from .group_ops import GroupOps
from .attribute_ops import AttributeOps

class DataContainer(GroupOps, DatasetOps, AttributeOps):
    """
    Manages and serializes data into HDF5 format.

    This class provides structured handling of groups and datasets read with the reader classes. It allows for easy access to the data and attributes stored in the DataContainer.
    It also provides functions for easy serialization and data visualization.
    """
    def __init__(self):
        """ Initializes the DataContainer with predefined groups and datasets.
        """
        super().__init__()

        # Set root node
        self.nodes["/"] = RootNode()

        # Set initial groups and datasets
        self.add_group("/", "Data")
        self.add_group("/", "GroundTruth")
        self.add_group("/", "MetaData")
        self.add_dataset("/Data", "Tdata")
        self.add_dataset("/GroundTruth", "DefectMask")
        self.add_dataset("/MetaData", "LookUpTable")
        self.add_dataset("/MetaData", "ExcitationSignal")
        self.add_dataset("/MetaData", "DomainValues")

   # Overwrite the __str__ method to provide a string representation of the DataContainer
    def __str__(self):
        returnstring = ""
        for path, node in self.nodes.items():
            returnstring = returnstring + f"{path}: ({node.name}: {node.type})" + "\n"

        return returnstring

