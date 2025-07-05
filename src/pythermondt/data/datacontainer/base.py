from collections.abc import ItemsView, KeysView, ValuesView
from sys import getsizeof
from typing import TypeVar, overload

from .node import DataNode, GroupNode, RootNode
from .utils import format_bytes, split_path

# Add typevars to make type hinting more readable
NodeTypes = RootNode | GroupNode | DataNode
T = TypeVar("T", bound=NodeTypes)


# Custom class to access nodes in the DataContainer ==> Needed to make the node property indexable and callable
class NodeAccessor:
    def __init__(self):
        self.__nodes: dict[str, NodeTypes] = {}

    # Functions to get views of the __nodes dictionary
    def keys(self) -> KeysView[str]:
        return self.__nodes.keys()

    def values(self) -> ValuesView[NodeTypes]:
        return self.__nodes.values()

    def items(self) -> ItemsView[str, NodeTypes]:
        return self.__nodes.items()

    # Private methods to get, set and delete nodes
    @overload
    def __get_node(self, key: str) -> NodeTypes: ...

    @overload
    def __get_node(self, key: str, *node_types: type[T]) -> T: ...

    def __get_node(self, key: str, *node_types: type[T]) -> NodeTypes | T:
        # Check if the path exists
        if key not in self.__nodes:
            raise KeyError(f"Node at path '{key}' does not exist.")

        # Optionally check if the node is of the correct type
        node = self.__nodes[key]
        if node_types and not isinstance(node, node_types):
            raise TypeError(f"Node at path '{key}' is not of type: {', '.join([t.__name__ for t in node_types])}.")
        return node

    def __set_node(self, key: str, value: NodeTypes) -> None:
        # Block any overwrites by default ==> updating nodes is handled using __get_node
        # by overriding .data attribute in Node classes
        if key in self.__nodes:
            raise KeyError(f"Node at path '{key}' already exists. Use a different path or delete the existing node.")

        # Special handling for root nodes
        if isinstance(value, RootNode):
            if key != "/":
                raise ValueError("RootNode must be placed at the root path '/'.")

            if "/" in self.__nodes:
                raise ValueError(
                    "RootNode already exists in the DataContainer. RootNode must be unique for each DataContainer."
                )

            self.__nodes[key] = value
            return

        # Split path in parent and child and check if parent exists
        parent, _ = split_path(key)
        if parent not in self.__nodes:
            if parent == "/" and parent not in self.__nodes:
                raise KeyError(
                    "RootNode does not exist in this container. "
                    "Check container initialization and ensure that a RootNode exists!"
                )
            else:
                raise KeyError(f"Parent node at path '{parent}' does not exist.")

        # Also check if parent is a RootNode or GroupNode
        if not isinstance(self.__nodes[parent], (RootNode, GroupNode)):
            raise TypeError(f"Parent node at path '{parent}' must be a RootNode or GroupNode.")

        # Set node at path in dictionary
        self.__nodes[key] = value

    def __delete_node(self, key: str) -> None:
        # Check if the path exists
        if key not in self.__nodes:
            raise KeyError(f"Node at path '{key}' does not exist.")

        # If it's a group, remove all child nodes as well to avoid orphaned nodes
        if isinstance(self.__nodes[key], GroupNode):
            child_keys = [k for k in self.__nodes.keys() if k.startswith(key + "/")]
            for child_key in child_keys:
                del self.__nodes[child_key]

        del self.__nodes[key]

    # Overriding the getitem, setitem and delitem methods to access nodes ==> Makes the node property indexable
    def __getitem__(self, key: str) -> NodeTypes:
        return self.__get_node(key)

    def __delitem__(self, key: str) -> None:
        self.__delete_node(key)

    def __setitem__(self, key: str, value: NodeTypes):
        self.__set_node(key, value)

    # Overriding the call method to access nodes ==> Makes the node property callable like a function
    @overload
    def __call__(self, key: str) -> NodeTypes: ...

    @overload
    def __call__(self, key: str, *node_types: type[T]) -> T: ...

    def __call__(self, key: str, *node_types: type[T]) -> NodeTypes | T:
        if not node_types:
            return self.__get_node(key)
        return self.__get_node(key, *node_types)


class DataContainerBase:
    def __init__(self):
        if type(self) is DataContainerBase:
            raise TypeError(f"{self.__class__.__name__} is a base class and cannot be instantiated directly.")
        self.__node_accessor = NodeAccessor()

    @property
    def nodes(self) -> NodeAccessor:
        """Property to access nodes in the DataContainer.

        This property is indexable and callable to get and set nodes. The path is checked for existence. Optionally the
        nodes is also checked for the correct type (provided as a function argument) while getting a node. Therefore the
        returned type is ensured to be correct. When setting nodes, the parent path is checked for existence and type
        (either RootNode or GroupNode). It also allows for deleting nodes by using the del keyword. If a group is
        deleted, all child nodes are also deleted to avoid orphaned nodes. Use with caution! Some sanity checks are
        disabled when using the property directly. Only for advanced users!

        Example usage:
        ```python
        from pythermondt.data import DataContainer
        from pythermondt.data.datacontainer.node import GroupNode, DataNode

        # Get nodes
        # Returns the node at path "/Data"
        node = data_container.nodes("/Data")

        # Returns the node at path "/Data". Raises TypeError if the node is not a DataNode
        node = data_container.nodes("/Data", DataNode)

        # Returns the node at path "/Data". Raises TypeError if the node is not a DataNode or GroupNode
        node = data_container.nodes("/Data", DataNode, GroupNode)

        # Set nodes
        data_container.nodes["/Data"] = DataNode("name", data)
        data_container.nodes["/Data"] = GroupNode("name")

        # Delete a node
        del data_container.nodes["/Data"]
        ```
        """
        return self.__node_accessor


# Basic operations for the DataContainer
class BaseOps(DataContainerBase):
    def _path_exists(self, key: str) -> bool:
        """Check if a path exists in the DataContainer.

        Parameters:
            key (str): The path to check.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        try:
            self.nodes(key)
            return True
        except KeyError:
            return False

    def _is_datanode(self, key: str) -> bool:
        """Check if a DataNode exists at the given path.

        Parameters:
            key (str): The path to check.

        Returns:
            bool: True if a DataNode exists at the path, False otherwise.
        """
        try:
            self.nodes(key, DataNode)
            return True
        except (KeyError, TypeError):
            return False

    def _is_groupnode(self, key: str) -> bool:
        """Check if a GroupNode exists at the given path.

        Parameters:
            key (str): The path to check.

        Returns:
            bool: True if a GroupNode exists at the path, False otherwise.
        """
        try:
            self.nodes(key, GroupNode)
            return True
        except (KeyError, TypeError):
            return False

    def _is_rootnode(self, key: str) -> bool:
        """Check if a RootNode exists at the given path.

        Parameters:
            key (str): The path to check.

        Returns:
            bool: True if the given path is the root path, False otherwise.
        """
        try:
            self.nodes(key, RootNode)
            return True
        except (KeyError, TypeError):
            return False

    def _parent_exists(self, key: str) -> bool:
        """Check if the parent of the given path exists and is a GroupNode or RootNode.

        If the path itself is the root path, it returns False.

        Parameters:
            key (str): The path to check.

        Returns:
            bool: True if the parent exists and is a GroupNode or RootNode, False otherwise.
        """
        # Return false if the actual path is the root path
        if self._is_rootnode(key):
            return False

        parent, _ = split_path(key)
        try:
            self.nodes(parent, GroupNode, RootNode)
            return True
        except (KeyError, TypeError):
            return False

    def memory_bytes(self) -> int:
        """Returns the total memory size of the DataContainer in bytes.

        This includes the memory size of all nodes and their attributes.
        """
        return getsizeof(self) + getsizeof(self.nodes) + sum(n.memory_bytes() for n in self.nodes.values())

    def print_memory_usage(self):
        """Print detailed memory usage breakdown by path."""
        total_bytes = self.memory_bytes()

        print(f"DataContainer Memory Usage: {format_bytes(total_bytes)}")
        print("-" * len(f"DataContainer Memory Usage: {format_bytes(total_bytes)}"))

        # Collect node sizes
        node_sizes = []
        for path, node in self.nodes.items():
            size_bytes = node.memory_bytes()
            percentage = (size_bytes / total_bytes) * 100
            node_sizes.append((path, size_bytes, percentage, node.type.value))

        # Sort by size (largest first)
        node_sizes.sort(key=lambda x: x[1], reverse=True)

        # Print breakdown
        for path, size_bytes, percentage, node_type in node_sizes:
            size_str = format_bytes(size_bytes)
            print(f"{path:<30} {size_str:>10} ({percentage:>5.1f}%) [{node_type}]")
