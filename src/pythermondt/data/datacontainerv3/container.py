from typing import Dict
from .node import Node, RootNode, GroupNode, DataNode, NodeType
from .utils import validate_path


class DataContainer:
    def __init__(self):
        self._nodes: Dict[str, Node] = {}

        # Set root node
        self._nodes["/"] = RootNode()

        # Set initial groups
        self.__add_group("/", "Data")
        self.__add_group("/", "GroundTruth")
        self.__add_group("/", "MetaData")

        # Set initial datasets
        self.__add_dataset("/Data1", "Temperature")

    def __str__(self):
        return f"DataContainer with {len(self._nodes)} nodes."

    def __add_group(self, path: str, name):
        # Generate key
        key = path + name

        if key in self._nodes.keys():
            raise KeyError(f"Group with name: '{name}' at the path: '{path}' already exists.")
        node = GroupNode(name)
        self._nodes[key] = GroupNode(name)

    def __add_dataset(self, path: str, name):
        # Generate key
        key = path + name

        # Check if the node at the given path already exists and is a group or root node
        if path not in self._nodes.keys():
            raise KeyError(f"Group at path: '{path}' does not exist.")
        
        if self._nodes[path].type != NodeType.GROUP and self._nodes[path].type != NodeType.ROOT:
            raise KeyError(f"Node at path: '{path}' is not a group or root node. Cannot add dataset.")

        if key in self._nodes.keys():
            raise KeyError(f"Dataset with name: '{name}' at the path: '{path}' already exists.")
        self._nodes[key] = DataNode(name)