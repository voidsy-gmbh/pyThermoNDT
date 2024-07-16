import torch
from torch.utils.data import Dataset
from .data_container import DataContainer
from ._base_reader import _BaseReader

class ThermoDataset(Dataset):
    def __init__( self, data_reader: _BaseReader, input_dataset_path: str, label_dataset_path: str, transform=None):
        """
        Initialize the PyTorch dataset with the data reader and the input and label groups and datasets.

        Parameters:
        - data_reader (BaseReader): The data reader object to use for loading data.
        - input_dataset_path (str): The path to the input dataset in the DataContainer. Should be in the format 'group/dataset'.
        - label_dataset_path (str): The path to the label dataset in the DataContainer. Should be in the format 'group/dataset'.
        - transform (callable, optional): Optional transform to be applied to the input data.
        """
        # Read Variables
        self.data_reader = data_reader
        self.input_dataset_path = input_dataset_path
        self.label_dataset_path = label_dataset_path
        self.transform = transform

        # Get file paths
        self._file_paths = self.data_reader.file_paths()
    
    def __len__(self):
        # We use the number of of files that the data_reader can read
        return self.data_reader.num_files
    
    def __getitem__(self, idx):
        # Load one datapoint as a DataContainer using the reader
        datapoint = self.data_reader[idx]

        # Check if the DataContainer is valid
        if not isinstance(datapoint, DataContainer):
            raise ValueError(f"DataContainer is not valid. Got {type(datapoint)}")
        
        # Apply the transform to the input data if it is not None
        # TODO: Implement the transforms
        # Input and Output Datacontainer
        if self.transform:
            NotImplementedError("Applying transforms is not implemented yet")

        # Get the input and label data from the DataContainer
        input_data = datapoint.get_dataset_from_path(self.input_dataset_path)
        label_data = datapoint.get_dataset_from_path(self.label_dataset_path)

        # Convert numpy arrays to torch tensors
        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(label_data)

        return input_tensor, label_tensor