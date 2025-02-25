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
        key, _, _ = generate_key(path, name)
        self.nodes[key] = GroupNode(name)

    def get_all_groups(self) -> list[str]:
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
        # Only remove the group if it is a GroupNode
        if not self._is_groupnode(path):
            raise KeyError(f"Group with path: '{path}' does not exist or is not a group.")
        del self.nodes[path]
