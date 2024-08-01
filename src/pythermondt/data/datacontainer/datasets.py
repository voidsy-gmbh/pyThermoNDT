import torch
from typing import List
from torch import Tensor
from numpy import ndarray
from .interface import _IDdataContainer

class Datasets(_IDdataContainer):
    def _add_dataset(self, group_name: str, dataset_name: str, data: Tensor | ndarray | None = None, **attributes):
        """
        Adds an empty dataset to a specified group within the DataContainer. Optionally, initial data and attributes can be provided.

        Parameters:
        - group_name (str): The name of the group to add the dataset to.
        - dataset_name (str): The name of the dataset to add.
        - data (np.array, optional): Initial data for the dataset. Defaults to None.
        - **attributes: Optional attributes to add to the dataset as key-value pairs.
        """
        if group_name not in self._groups:
            raise ValueError("This group does not exist")
        
        # Convert to torch.Tensor if np.ndarray
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)
        
        self._datasets[(group_name, dataset_name)] = data
        self._attributes[(group_name, dataset_name)] = attributes

    def _add_datasets(self, group_name, dataset_names, data: Tensor | ndarray | None = None):
        """
        Adds a set of emtpy datasets to a specified group within the DataContainer.

        Parameters:
        - group_name (str): The name of the group to add the datasets to.
        - dataset_names (List[str]): The list of dataset names to add.
        - data (np.array, optional): Initial data for the datasets. Defaults to None.
        - **attributes: Optional attributes to add to the datasets as key-value pairs.
        """
        if group_name not in self._groups:
            raise ValueError("This group does not exist")

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        # Convert to torch.Tensor if np.ndarray
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)
        
        for dataset_name in dataset_names:
            self._datasets[(group_name, dataset_name)] = data
            self._attributes[(group_name, dataset_name)] = {}

    def get_datasets(self, group_name: str='') -> List[str]:
        '''
        Get a list of all datasets in a group or in the DataCotainer

        Parameters:
        - group_name (str): The name of the group to get the datasets from. If not specified, all datasets in the DataContainer are returned.

        Returns:
        - List[str]: A list of all datasets in the specified group or in the DataContainer.
        '''
        if group_name == '':
            return [dataset for _ , dataset in self._datasets.keys()]
        else:
            return [dataset for group, dataset in self._datasets.keys() if group == group_name]
        
    def get_dataset(self, path: str) -> Tensor:
        """
        This method allows for direct access to the underlying data in a controlled manner, ensuring that data retrieval is both predictable and error-resistant. 
        It supports modular access to various datasets for processing and analysis. Retrieves a dataset by specifying its path.

        Parameters:
        - path (str): The path to the dataset in the form of 'group_name/dataset_name'.

        Returns:
        - torch.Tensor | str: The data stored in the specified dataset.
        """
        # Split the path into group and dataset names
        group_name, dataset_name = path.split('/') if '/' in path else (path, '')

        # Check path
        if dataset_name == '':
            raise ValueError("The provided path does not contain a dataset name or is not in the form 'group_name/dataset_name'.")
        if group_name not in self._groups:
            raise ValueError(f"The group {group_name} does not exist.")
        if (group_name, dataset_name) not in self._datasets:
            raise ValueError(f"The dataset {dataset_name} in group {group_name} does not exist.")

        # Return Dataset if it exists
        dataset = self._datasets[(group_name, dataset_name)]

        if dataset is None:
            return torch.empty(0)
        else:
            return dataset
    
    def fill_dataset(self, path: str, data: Tensor | ndarray, **attributes):
        """
        Fills specified dataset with data and updates the attributes.

        Parameters:
        - path (str): The path to the dataset in the form of 'group_name/dataset_name'.
        - data (np.ndarray | torch.Tensor | str): The data to fill the dataset with. np.ndarrays are automatically converted to torch.Tensors.
        - **attributes: Optional attributes to add to the dataset as key-value pairs.
        """
        # Split the path into group and dataset names
        group_name, dataset_name = path.split('/') if '/' in path else (path, '')

        # Check path
        if dataset_name == '':
            raise ValueError("The provided path does not contain a dataset name or is not in the form 'group_name/dataset_name'.")
        if group_name not in self._groups:
            raise ValueError(f"The group {group_name} does not exist.")
        if (group_name, dataset_name) not in self._datasets:
            raise ValueError(f"The dataset {dataset_name} in group {group_name} does not exist.")
        
        # Convert to torch.Tensor if np.ndarray
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)
        
        self._datasets[(group_name, dataset_name)] = data
        self.update_attributes(path="/".join([group_name, dataset_name]), **attributes)