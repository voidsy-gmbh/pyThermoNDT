from .groups import Groups
from .datasets import Datasets
from .attributes import Attributes

class DataContainer(Groups, Datasets, Attributes):
    """
    Manages and serializes data into HDF5 format.

    This class provides structured handling of groups and datasets read with the reader classes. It allows for easy access to the data and attributes stored in the DataContainer.
    It also provieds functions for easy serialization and data visualization.
    """
    def __init__(self):
        """
        Initializes the DataContainer with predefined groups and datasets.
        """
        # Initialize an empty DataContainer
        # Define the structure of the DataContainer: Groups
        self._add_group(['Data', 'GroundTruth', 'MetaData'])

        # Define the structure of the DataContainer: Datasets
        self._add_datasets(group_name='GroundTruth', dataset_names='DefectMask')
        self._add_datasets(group_name='Data', dataset_names='Tdata')
        self._add_datasets(group_name='MetaData', dataset_names=['LookUpTable', 'ExcitationSignal', 'DomainValues'])

    # Override string method for nicer output
    def __str__(self):
        # This method will return a string representation of the DataContainer
        groups_info = ", ".join(self._groups)  # A simple string listing all groups
        datasets_info = ", ".join(f"{group}/{dataset}" for group, dataset in self._datasets.keys())  # List all datasets by group/dataset pair
        return f"\nDataContainer with:\nGroups: {groups_info}\nDatasets: {datasets_info} \n"