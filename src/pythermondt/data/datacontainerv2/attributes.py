from typing import Tuple, Dict
from .path_check import validate_path

class _Attributes:
    def __init__(self):
        self.__attributes: dict = {}

    def add_attribute(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """ Adds attributes to a dataset or a group specified by the path.

        Parameters:
            path (str): The path to the group or dataset in the from of 'group_name/dataset_name'. 
                In case of a group, the path should be just the group name.
            **attributes (Dict[str, str | int | float | list | dict]): The attributes to add as key-value pairs.
        """
        # Validate the path
        path = validate_path(path)

        for name, value in attributes.items():
            # Check if the attribute already exists
            if name in self.__attributes[path]:
                raise ValueError(f"Attribute {name} already exists for the path {path}. Consider using the update_attributes method to update the attribute value")
            
            # Create a new attribute
            self.__attributes[path][name] = value

    def get_attribute(self, path: str, attribute_name: str) -> str | int | float | list | dict:
        """ Retrieves an attribute from a dataset or a group specified by the path.

        Parameters:
            path (str): The path to the group or dataset in the from of 'group_name/dataset_name'. 
                In case of a group, the path should be just the group name.
            attribute_name (str): The name of the attribute to retrieve.
        """
        # Validate the path
        path = validate_path(path)
        
        # Check if the path exists ==> Check if dataset or group exists
        if path not in self.__attributes:
            raise ValueError(f"The path {path} does not exist.")
        
        return self.__attributes[path][attribute_name]
        
    def update_attributes(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """ Updates attributes for a group or dataset.

        Parameters:
            path (str): The path to the group or dataset in the from of 'group_name/dataset_name'. 
                In case of a group, the path should be just the group name.
            **attributes (Dict[str, str | int | float | list | dict]): The attributes to update as key-value pairs.
        """
        # Validate the pat
        path = validate_path(path)

        # Check if the path exists ==> Check if dataset or group exists
        if path not in self.__attributes:
            raise ValueError(f"The path {path} does not exist.")
        
        # Check if the attribute already exists ==> if the attribute does not exist, create it using the add_attribute method
        for name, value in attributes.items():
            if name not in self.__attributes[path]:
                self.add_attribute(path, name=value)
        
        # Update the attributes
        self.__attributes[path].update(attributes)