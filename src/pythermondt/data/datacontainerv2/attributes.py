from typing import Tuple, Dict

class _Attributes:
    def __init__(self):
        self._attributes: dict = {}

    def add_attribute(self, key: Tuple[str, str], **attributes: Dict[str, str | int | float | list | dict]):
        """
        Adds attributes to a dataset or a group specified by the key.

        Parameters:
        -----------
        - key (Tuple[str, str]): The key to the group or dataset in the form of ('group_name', 'dataset_name'). In case 
                                of a group, the dataset_name should be an empty string.
        - **attributes: The attributes to add as key-value pairs.
        """
        for name, value in attributes.items():
            # Check if the attribute already exists
            if name in self._attributes[key]:
                raise ValueError(f"Attribute {name} already exists for the key {key}. Consider using the update_attributes method to update the attribute value")
            
            # Create a new attribute
            self._attributes[key][name] = value

    def get_attribute(self, key: Tuple[str, str], attribute_name: str) -> str | int | float | list | dict:
        """
        Retrieves an attribute from a dataset or a group specified by the key.

        Parameters:
        -----------
        - key (Tuple[str, str]): The key to the group or dataset in the form of ('group_name', 'dataset_name'). In case 
                                of a group, the dataset_name should be an empty string.
        - attribute_name (str): The name of the attribute to retrieve.

        Returns:
        ---------------
        - str | int | float | list | dict: The attribute value.
        """
        # Check if the key exists ==> Check if dataset or group exists
        if key not in self._attributes:
            raise ValueError(f"The key {key} does not exist.")
        
        return self._attributes[key][attribute_name]
        
    def update_attributes(self, key: Tuple[str, str], **attributes: Dict[str, str | int | float | list | dict]):
        """
        Updates attributes for a group or dataset.

        Parameters:
        - key (Tuple[str, str]): The key to the group or dataset in the form of ('group_name', 'dataset_name'). In case
                                of a group, the dataset_name should be an empty string.
        - **attributes: The attributes to update as key-value pairs.
        """
        # Check if the key exists ==> Check if dataset or group exists
        if key not in self._attributes:
            raise ValueError(f"The key {key} does not exist.")
        
        # Check if the attribute already exists ==> if the attribute does not exist, create it using the add_attribute method
        for name, value in attributes.items():
            if name not in self._attributes[key]:
                self.add_attribute(key, name=value)
        
        # Update the attributes
        self._attributes[key].update(attributes)