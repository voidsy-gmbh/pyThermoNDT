from typing import List
from torch import Tensor
from numpy import ndarray
from .interface import _IDdataContainer

class Attributes(_IDdataContainer):
    def get_attribute(self, path: str, attribute_name: str) -> str | int | float | list | dict:
        """
        Retrieves an attribute from a dataset or a group specified by the path.

        Parameters:
        - path (str): The path to the dataset in the form of 'group_name/dataset_name'.
        - attribute_name (str): The name of the attribute to retrieve.

        Returns:
        - str | int | float | list | dict: The attribute value.
        """
        # Split the path into group and dataset names
        group_name, dataset_name = path.split('/') if '/' in path else (path, '')

        # Check path
        if group_name not in self._groups:
            raise ValueError(f"The group {group_name} does not exist.")
        if (group_name, dataset_name) not in self._datasets and dataset_name != '':
            raise ValueError(f"The dataset {dataset_name} in group {group_name} does not exist.")
        
        # If the path is a group, return the attribute from the group
        if group_name != '' and dataset_name == '':
            attributes = self._attributes[group_name]

            # Check if the attribute exists
            if attribute_name not in attributes:
                raise ValueError(f"The attribute {attribute_name} does not exist in group {group_name}.")
            return attributes[attribute_name]
            
        # If the path is a dataset, return the attribute from the dataset
        elif group_name != '' and dataset_name != '':
            attributes = self._attributes[(group_name, dataset_name)]

            # Check if the attribute exists
            if attribute_name not in attributes:
                raise ValueError(f"The attribute {attribute_name} does not exist in dataset {dataset_name} of group {group_name}.")
            return attributes[attribute_name]
        
        # Else, the path is invalid
        else:
            raise ValueError(f"The provided path: {path} is not valid.")
        
    def update_attributes(self, path: str, **attributes):
        """
        Updates attributes for a group or dataset.

        Parameters:
        - path (str): The path to the group or dataset in the form of 'group_name/dataset_name'.
        - **attributes: The attributes to update as key-value pairs.
        """
        # Split the path into group and dataset names
        group_name, dataset_name = path.split('/') if '/' in path else (path, '')

        # Check path
        if group_name not in self._groups:
            raise ValueError(f"The group {group_name} does not exist.")
        if (group_name, dataset_name) not in self._datasets and dataset_name != "":
            raise ValueError(f"The dataset {dataset_name} in group {group_name} does not exist.")

        if dataset_name=="":  # Update group attributes
            self._attributes[group_name].update(attributes)
        else: # Else update dataset attributes
            self._attributes[(group_name, dataset_name)].update(attributes)
