from torch.utils.data import Dataset
from .datacontainer import DataContainer
from ..readers import BaseReader

class ThermoDataset(Dataset):
    def __init__( self, data_reader: BaseReader, input_dataset_path: str, label_dataset_path: str):
        """
        Initialize the PyTorch dataset with the data reader and the input and label groups and datasets.

        Parameters:
            data_reader (BaseReader): The data reader object to use for loading data.
            input_dataset_path (str): The path to the input dataset in the DataContainer.
            label_dataset_path (str): The path to the label dataset in the DataContainer. 
        """
        # Read Variables
        self.data_reader = data_reader
        self.input_dataset_path = input_dataset_path
        self.label_dataset_path = label_dataset_path
    
    def __len__(self):
        # We use the number of of files that the data_reader can read
        return self.data_reader.num_files
    
    def __getitem__(self, idx):
        # Load one datapoint as a DataContainer using the reader
        datapoint = self.data_reader[idx]

        # Check at runtime if the datapoint is a DataContainer
        if not isinstance(datapoint, DataContainer):
            raise ValueError(f"DataContainer is not valid. Got {type(datapoint)}")

        # Get the input and label data from the DataContainer
        input_tensor = datapoint.get_dataset(self.input_dataset_path)
        label_tensor = datapoint.get_dataset(self.label_dataset_path)

        return input_tensor, label_tensor