from typing import List
from .interface import _IDdataContainer

class Groups(_IDdataContainer):
    def __add_group(self, group_names: str | List[str], **attributes):
        """
        Adds a single group or a list of group names to the DataContainer with optional attributes.

        Parameters:
        - group_names (str | List[str]): The name or list of names of the groups to add.
        - **attributes: Optional attributes to add to the groups as key-value pairs.
        """
        if isinstance(group_names, str):
            group_names = [group_names]
        
        for group_name in group_names:
            self._groups.append(group_name)
            self._attributes[group_name] = attributes

    def get_groups(self) -> List[str]:
        '''
        Get a list of all groups in the DataContainer.
        
        Returns:
        - List[str]: A list of all groups in the DataContainer.
        '''
        return self._groups.copy()