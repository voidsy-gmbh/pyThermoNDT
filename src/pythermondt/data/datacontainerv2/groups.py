from typing import List
from .attributes import _Attributes
from .path_check import validate_paths

class _Groups:
    def __init__(self):
        self.__groups: List[str] = []
        self.__attributes = _Attributes()

    def _add_group(self, group_names: str | List[str], **attributes):
        """ Adds a single group or a list of group names to the DataContainer with optional attributes.

        Parameters:
            group_names (str | List[str]): The name or list of names of the groups to add.
            **attributes (Dict[str, str | int | float | list | dict]): Optional attributes to add to the groups as key-value pairs. 
                Be careful, if you add a list of groups, the same attributes will be added to all groups!
        """
        # If group_names is a single string, convert it to a list
        group_names = [group_names] if isinstance(group_names, str) else group_names

        # Validate and normalize the group names
        group_names = validate_paths(group_names)

        # Check if the group already exists
        for group_name in group_names:           
            if group_name in self.__groups:
                raise ValueError(f"Group {group_name} already exists")
        
        # Add group and attributes
        for group_name in group_names:
            self.__groups.append(group_name)
            self.__attributes.add_attribute(group_name, **attributes)

    def get_groups(self) -> List[str]:
        ''' Get a list of all groups in the DataContainer.
        
        Returns:
            A list of all groups in the DataContainer.
        '''
        return [group[1:] for group in self.__groups.copy()]