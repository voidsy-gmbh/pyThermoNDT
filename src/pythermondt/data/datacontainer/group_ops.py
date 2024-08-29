from typing import List
from .base import BaseOps
from .node import GroupNode
from .utils import generate_key

class GroupOps(BaseOps):
    def add_group(self, path: str, name: str):
        """Adds a single group to a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.
            name (str): The name of the group to add.

        Raises:
            KeyError: If the parent group does not exist.
            KeyError: If the group already exists.
        """
        key, parent, child = generate_key(path, name)

        if self._is_groupnode(key):
            raise KeyError(f"Group with name: '{child}' at the path: '{parent}' already exists.")

        self.nodes[key] = GroupNode(name)

    def get_groups(self) -> List[str]:
        """Get a list of all groups in the DataContainer.

        Returns:
            List[str]: A list of all groups in the DataContainer.
        """
        return [node.name for node in self.nodes.values() if isinstance(node, GroupNode)]
    
    def remove_group(self, path: str):
        """Removes a single group from a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.

        Raises:
            KeyError: If the group does not exist.
        """
        del self.nodes[path]