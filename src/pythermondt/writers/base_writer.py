from abc import ABC, abstractmethod
from ..data import DataContainer

class BaseWriter(ABC):
    @abstractmethod
    def __init__(self):
        """ Constructor for the BaseWriter class. Should be called by all subclasses. """
        pass
    
    @abstractmethod
    def write(self, container: DataContainer, file_name: str):
        """Actual implementation of the writing a single DataContainer to the destination folder.
        
        Parameters:
            container (DataContainer): The DataContainer which should be written to the destination folder.
            destination_folder (str): The destination folder to write to.
            file_name (str): The name of the DataContainer.
        """
        pass