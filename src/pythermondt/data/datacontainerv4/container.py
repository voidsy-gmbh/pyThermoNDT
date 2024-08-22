from typing import Dict ,Callable
from functools import wraps
from .node import Node, RootNode
from .dataset_ops import DatasetOps
from .group_ops import GroupOps
from .attribute_ops import AttributeOps

class DataContainer:
    """
    Manages and serializes data into HDF5 format.

    This class provides structured handling of groups and datasets read with the reader classes. It allows for easy access to the data and attributes stored in the DataContainer.
    It also provides functions for easy serialization and data visualization.
    """
    def __init__(self):
        """ Initializes the DataContainer with predefined groups and datasets.
        """
        self.__nodes: Dict[str, Node] = {}
        self.__nodes["/"] = RootNode()

        self.__ops = {
            'dataset': DatasetOps(self.__nodes),
            'group': GroupOps(self.__nodes),
            'attribute': AttributeOps(self.__nodes),
        }

        # Set initial groups
        self.__ops['group'].add_group("/", "Data")
        self.__ops['group'].add_group("/", "GroundTruth")
        self.__ops['group'].add_group("/", "GroundTruth")
        self.__ops['group'].add_group("/", "MetaData")

        # Set initial datasets
        self.__ops['dataset'].add_dataset("/Data", "Tdata")
        self.__ops['dataset'].add_dataset("/GroundTruth", "DefectMask")
        self.__ops['dataset'].add_dataset("/MetaData", "LookUpTable")
        self.__ops['dataset'].add_dataset("/MetaData", "ExcitationSignal")
        self.__ops['dataset'].add_dataset("/MetaData", "DomainValues")

    def __str__(self):
        return "\n".join(f"{path}: ({node.name}: {node.type})" for path, node in self.__nodes.items())

    def __getattr__(self, name):
        # List of restricted method names
        restricted_methods = ['add_group', 'add_dataset']

        if name in restricted_methods:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'. This method is restricted.")

        for ops in self.__ops.values():
            if hasattr(ops, name):
                return getattr(ops, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")