from abc import ABC
from typing import Dict, TypeVar,Optional, Type
from .utils import split_path
from .node import RootNode, GroupNode, DataNode

# Add typevars to make type hinting more readable
NodeTypes = RootNode | GroupNode | DataNode
T = TypeVar("T", bound=NodeTypes)

class DataContainerBase(ABC):
    def __init__(self):
        self.__nodes: Dict[str, NodeTypes] = {}

    # Properties to access keys and values
    @property
    def keys(self):
        return self.__nodes.keys()
    
    @property
    def values(self):
        return self.__nodes.values()
    
    @property
    def items(self):
        return self.__nodes.items()

    # Private methods to get and set nodes
    def _getnode(self, key: str, node_type: Optional[Type[T]] = None) -> NodeTypes:
        # Check if path exists
        if key not in self.__nodes:
            raise KeyError(f"Node at path '{key}' does not exist.")
        
        # Optionally check if node is of the correct type
        node = self.__nodes[key]
        if node_type is not None and not isinstance(node, node_type):
            raise TypeError(f"Node at path '{key}' is not of type {node_type.__name__}.")
        return node
    
    def _setnode(self, key: str, value: NodeTypes) -> None:
        # Split path in parent and child and check if parent exists
        parent, _ = split_path(key)
        if parent != "/" and parent not in self.__nodes:
            raise KeyError(f"Parent node at path '{parent}' does not exist.")
        
        # Also check if parent is a RootNode or GroupNode
        if parent != "/" and not isinstance(self.__nodes[parent], (RootNode, GroupNode)):
            raise TypeError(f"Parent node at path '{parent}' must be a RootNode or GroupNode.")
        
        # Set node at path in dictionary
        self.__nodes[key] = value

    def _removenode(self, key: str) -> None:
        if key not in self.__nodes:
            raise KeyError(f"Node at path '{key}' does not exist.")
        
        # If it's a group, remove all child nodes as well to avoid orphaned nodes
        if isinstance(self.__nodes[key], GroupNode):
            child_keys = [k for k in self.__nodes.keys() if k.startswith(key + '/')]
            for child_key in child_keys:
                del self.__nodes[child_key]
        
        del self.__nodes[key]

    # Inner class to make the node property indexable
    class __NodeAccessor:
        def __init__(self, outer):
            self.__outer = outer

        def __getitem__(self, key: str) -> NodeTypes:
            return DataContainerBase._getnode(self.__outer,key)
        
        def __delitem__(self, key: str) -> None:
            DataContainerBase._removenode(self.__outer, key)

        def __setitem__(self, key: str, value: NodeTypes):
            DataContainerBase._setnode(self.__outer, key, value)

        def __call__(self, key: str, node_type: Optional[Type[T]] = None) -> NodeTypes:
            return DataContainerBase._getnode(self.__outer, key, node_type)   
    
    @property
    def nodes(self):
        """ Property to access nodes in the DataContainer.

        This property is indexable and callable to get and set nodes. The path is checked for existence. Optionally the nodes is also checked for the correct type for both 
        getting and setting nodes. When setting nodes, the parent path is also checked for existence and type (either RootNode or GroupNode). 
        It also allows for deleting nodes by using the del keyword. If a group is deleted, all child nodes are also deleted to avoid orphaned nodes.
        """
        return self.__NodeAccessor(self)
    
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

        Parameters:
            key (str): The path to check.

        Returns:
            bool: True if the parent exists and is a GroupNode or RootNode, False otherwise.
        """
        parent, _ = split_path(key)
        return self._is_groupnode(parent) or self._is_rootnode(parent)