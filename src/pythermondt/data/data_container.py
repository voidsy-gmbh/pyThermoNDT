import io, json, os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import List, Dict, Tuple

class DataContainer:
    """
    Manages and serializes data into HDF5 format.

    This class provides structured handling of groups and datasets read with the reader classes. It allows for easy access to the data and attributes stored in the DataContainer.
    It also provieds functions for easy serialization and data visualization.
    """
    def __init__(self):
        """
        Initializes the DataContainer with predefined groups and datasets.
        """
        self._groups = []
        self._datasets = {}
        self._attributes = {}

        # Define the structure of the DataContainer: Groups
        self.__add_group(['Data', 'GroundTruth', 'MetaData'])

        # Define the structure of the DataContainer: Datasets
        self.__add_datasets(group_name='GroundTruth', dataset_names='DefectMask')
        self.__add_datasets(group_name='Data', dataset_names='Tdata')
        self.__add_datasets(group_name='MetaData', dataset_names=['LookUpTable', 'SimulationParameter', 'ExcitationSignal', 'DomainValues'])

    # Override string method for nicer output
    def __str__(self):
        # This method will return a string representation of the DataContainer
        groups_info = ", ".join(self._groups)  # A simple string listing all groups
        datasets_info = ", ".join(f"{group}/{dataset}" for group, dataset in self._datasets.keys())  # List all datasets by group/dataset pair
        return f"\nDataContainer with:\nGroups: {groups_info}\nDatasets: {datasets_info} \n"

    def __add_group(self, group_names: str | List[str], **attributes):
        """
        Adds a single group or a list of group names to the DataContainer with optional attributes.

        Parameters:
        - group_names (str | List[str]): The name or list of names of the groups to add.
        - **attributes: Optional attributes to add to the groups as key-value pairs.
        """
        if isinstance(group_names, str):
            group_names = [group_names]
        
        for group_name in group_names:
            self._groups.append(group_name)
            self._attributes[group_name] = attributes

    def __add_dataset(self, group_name: str, dataset_name: str, data: Tensor | ndarray | str | None = None, **attributes):
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
        
        self._datasets[(group_name, dataset_name)] = data
        self._attributes[(group_name, dataset_name)] = attributes

    def __add_datasets(self, group_name, dataset_names, data: Tensor | ndarray | str | None = None):
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
        
        for dataset_name in dataset_names:
            self._datasets[(group_name, dataset_name)] = data
            self._attributes[(group_name, dataset_name)] = {}
        
    def get_dataset(self, path: str) -> Tensor | str:
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
        return self._datasets[(group_name, dataset_name)]
    
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
        if (group_name, dataset_name) not in self._datasets:
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

    def fill_dataset(self, path: str, data: Tensor | ndarray | str, **attributes):
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

    def save2hdf5(self, path):
        """
        First it serializes the DataContainer instance to an HDF5 file and then saves it to the specified path.

        Parameters:
        - path (str): The path where the HDF5 file should be saved.
        """
        # Extract directory path
        directory = os.path.dirname(path)
        
        # Check if the directory exists
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"The specified path {directory} is not a directory.")

        # Check if the filename has a .hdf5 extension
        if not path.endswith(('.hdf5', 'h5')):
            raise ValueError("The file must be an HDF5 file with a .hdf5 or .h5 extension.")
        
        with open(path, 'wb') as file:
            file.write(self.serialize().getvalue())
        
    def serialize(self) -> io.BytesIO:
        """
        Serializes the DataContainer instance to an HDF5 file.

        Returns:
        - io.BytesIO: The serialized DataContainer instance to a BytesIO object.
        """
        # Serialize the class instance into a BytesIO object
        hdf5_bytes = io.BytesIO()

        with h5py.File(hdf5_bytes, 'w') as f:
            # 1.) Create all the groups
            for  group in self._groups:
                # Create the group
                grp = f.create_group(group)

                # Add attributes to group if they exist
                if group in self._attributes and len(self._attributes[group]) != 0:
                    for attribute_name, value in self._attributes[group].items():
                        if isinstance(value, (list, tuple, dict)):
                            value = json.dumps(value)
                        grp.attrs.create(attribute_name, value)
            
            # 2.) Create all the datasets and add the attributes
            for dataset_name, data in self._datasets.items():
                # Create the dataset
                # Convert to numpy array before
                # print(dataset_name, f",Data: {data}", f",Datatype: {type(data)}")
                if data is None:
                    continue
                data = data.numpy(force=True) if isinstance(data, Tensor) else data

                # If the data is not scalar ==> apply compression, else leave the data as it is
                if not np.isscalar(data):
                    dset = f.create_dataset('/'.join(dataset_name), data=data, compression="gzip", compression_opts=9)
                else:
                    dset = f.create_dataset('/'.join(dataset_name), data=data)

                # Add attributes to dataset if they exist
                if dataset_name in self._attributes and len(self._attributes[dataset_name]) != 0:
                    for attribute_name, value in self._attributes[dataset_name].items():
                        # print(attribute_name, f",Data: {value}", f",Datatype: {type(value)}")
                        if value is None:
                            continue
                        if isinstance(value, (list, tuple, dict)):
                            value = json.dumps(value)
                        dset.attrs.create(attribute_name, value)

        # Rewind the buffer
        hdf5_bytes.seek(0)
        return hdf5_bytes

    def show_frame(self, frame_number: int, option: str="", subtract_tinit: str="SubstractTinit", cmap: str = 'plasma'):
        """
        Visualize a specific frame from the dataset with optional ground truth visualization and color mapping.

        Parameters:
        - frame_number (int): The frame number to visualize.
        - option (str): The visualization option to apply. Options are "ShowGroundTruth", "OverlayGroundTruth", or an empty string. 
        - subtract_tinit (str): Controls the subtraction of the initial frame from the series to highlight changes. Options are "SubstractTinit" or "DontSubtractTinit".
        - cmap (str): The color map to use for the visualization. Defaults to 'plasma'.
        """
        # Clear current figure
        plt.clf()

        # Extract the data from the container
        lookuptable = self.get_dataset('MetaData/LookUpTable')
        data = self.get_dataset('Data/Tdata')
        groundtruth = self.get_dataset('GroundTruth/DefectMask')

        # Type check the data and ground truth
        if not isinstance(data, Tensor):
            raise ValueError("Dataset must be of type torch.Tensor")
        if not isinstance(groundtruth, Tensor):
            raise ValueError("Ground truth must be of type torch.Tensor")  

        # Convert data from torch.Tensor to numpy array for plotting with matplotlib
        data = data.numpy(force=True) # Force=True to ensure that the data is copied to the CPU

        # Extract the first raw frame
        firstrawframe = data[:, :, 2]
        
        # If Data has been scaled ==> Subtract Tinit does not make sense
        if lookuptable is not None and not np.isnan(lookuptable).any():
            subtract_tinit = "DontSubtractTinit"
        
        # Subtract firstrawframe from the data if requested
        if subtract_tinit == "DontSubtractTinit":
            data_to_show = data[:, :, frame_number]
        elif subtract_tinit == "SubstractTinit" and firstrawframe is not None:
            data_to_show = data[:, :, frame_number] - firstrawframe
        else:
            data_to_show = data[:, :, frame_number]
        
        # Process the option for ground truth
        if option == "ShowGroundTruth":
            plt.subplot(1, 2, 1)
            plt.imshow(data_to_show, aspect='auto', cmap=cmap)
            plt.title(f'Frame Number: {frame_number}')
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(groundtruth, aspect='auto')
            plt.title('Ground Truth')

        elif option == "OverlayGroundTruth":
            plt.imshow(data_to_show, aspect='auto', cmap=cmap)  # Display the original data
            plt.colorbar()  # Display the colorbar
            plt.title(f'Frame Number: {frame_number}')
            
            if groundtruth is not None:
                # Prepare the overlay
                binary_gt = groundtruth > 0  # Create a binary mask of the ground truth
                rows, cols = groundtruth.shape
                gt_overlay = np.zeros((rows, cols, 3))  # Initialize an all-zero RGB image for the overlay
                gt_overlay[:, :, 1] = binary_gt  # Apply green in the binary mask areas
                
                plt.imshow(gt_overlay, alpha=0.5)  # Display overlay with transparency

        else:  # Default case, just show the frame data possibly subtracting firstrawframe
            plt.imshow(data_to_show, aspect='auto', cmap=cmap)
            plt.title(f'Frame Number: {frame_number}')
            plt.colorbar()
        
        plt.show()

    def show_pixel_profile(self, pixel_pos_x: int, pixel_pos_y: int, option: str=""):
        """
        Plot the profile of a specific pixel across the dataset's domain values with an option for data adjustment. The X-axis of the plot is labeled according 
        to the domaintype attribute, reflecting the dataset's domain (e.g., time, frequency). The Y-axis is generically labeled as 'Temperature in K'.

        Parameters:
        - pixel_pos_x (int): The X-coordinate (column index) of the pixel. Must be within the dataset's second dimension range.
        - pixel_pos_y (int): The Y-coordinate (row index) of the pixel. Must be within the dataset's first dimension range.
        - option (str): Controls the subtraction of the initial pixel value from the series to highlight changes. Options are "DontSubtractTinit" 
            for using the raw data or an empty string "" to apply subtraction. Defaults to an empty string.
        """
        #Clear the current figure
        plt.clf()

        # Extract the data from the container
        lookuptable = self.get_dataset('MetaData/LookUpTable')
        data = self.get_dataset('Data/Tdata')
        domainvalues = self.get_dataset('MetaData/DomainValues')
        domaintype = self.get_attribute('MetaData/DomainValues', 'DomainType')

        # Emsure that domaintype is a string
        if not isinstance(domaintype, str):
            raise ValueError("DomainType must be of type str.")

        # Type check the data and ground truth
        if not isinstance(data, Tensor):
            raise ValueError("Dataset must be of type torch.Tensor")
        if not isinstance(domainvalues, Tensor):
            raise ValueError("domainvalues must be of type torch.Tensor")
        
        # Convert data from torch.Tensor to numpy array for plotting with matplotlib
        data = data.numpy(force=True) # Force=True to ensure that the data is copied to the CPU

        # Extract the first raw frame
        firstrawframe = data[:,:,2]

        # Validate pixel positions and option value
        if pixel_pos_x < 0 or pixel_pos_y < 0 or pixel_pos_x >= data.shape[1] or pixel_pos_y >= data.shape[0]:
            raise ValueError("Pixel positions must be within the range of data dimensions.")
        if option not in ["DontSubtractTinit", ""]:
            raise ValueError('Option must be "DontSubtractTinit" or an empty string.')
        
        # If Data has been scaled ==> Subtract Tinit does not make sense
        if lookuptable is not None and not np.isnan(lookuptable).any():
            option = "DontSubtractTinit"
        
        if option == "DontSubtractTinit":
            temperature_profile = data[pixel_pos_y, pixel_pos_x, :]
        else:
            if firstrawframe is not None:
                temperature_profile = data[pixel_pos_y, pixel_pos_x, :] - firstrawframe[pixel_pos_y, pixel_pos_x]
            else:
                temperature_profile = data[pixel_pos_y, pixel_pos_x, :]
        
        plt.plot(domainvalues, temperature_profile)
        plt.title(f'Profile of Pixel: {pixel_pos_x},{pixel_pos_y}')
        plt.xlabel(domaintype)
        plt.ylabel('Temperature in K')
        plt.show()