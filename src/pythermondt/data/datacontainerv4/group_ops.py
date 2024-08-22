from .base_ops import BaseOps
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

        if not self._parent_exists(key):
            raise KeyError(f"Group at the path: '{parent}' does not exist. Cannot add group, without parent group.")

        if self._group_exists(key):
            raise KeyError(f"Group with name: '{child}' at the path: '{parent}' already exists.")
        self._nodes[key] = GroupNode(name)