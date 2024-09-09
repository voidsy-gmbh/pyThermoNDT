from typing import List, Optional
from torch.utils.data import Dataset
from .datacontainer import DataContainer
from ..transforms import ThermoTransform

class ThermoDataset(Dataset):
    def __init__( self, data_source: List[str], transform: Optional[ThermoTransform] = None, cache_source: bool = False):
        """
        Initialize a PyTorch dataset with a list of data sources. The sources are used to read the data and create the dataset.
        The source can either be a local file path or a cloud storage path. The file path can either be an path to a single file, a directory or a regex pattern.
        Examples of data sources are:
            - /path/to/local/file
            - file:///path/to/local/file
            - s3://bucket-name/path/to/file

        Parameters:
            data_source (List[str]): List of data sources to be used in the dataset
            transform (ThermoTransform, optional): Optional transform to be directly applied to the data when it is read
            cache_source (bool, optional): If True, all the file paths are cached in memory. Therefore changes to the file sources will not be noticed at runtime. Default is False.
        """
        pass
    
    def __len__(self) -> int:
        raise NotImplementedError("TBD")
    
    def __getitem__(self, idx) -> DataContainer:
        raise NotImplementedError("TBD")