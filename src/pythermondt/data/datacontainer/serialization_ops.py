import io
import json
from collections.abc import ItemsView
from typing import Any

import h5py

from .attribute_ops import AttributeOps
from .base import BaseOps
from .dataset_ops import DatasetOps
from .group_ops import GroupOps
from .node import AttributeTypes, DataNode, GroupNode, RootNode
from .utils import validate_path


class SerializationOps(BaseOps):
    def serialize_to_hdf5(self) -> io.BytesIO:
        """Serializes the DataContainer instance to an HDF5 file.

        Returns:
            io.BytesIO: The serialized DataContainer instance as a BytesIO object.
        """
        hdf5_bytes = io.BytesIO()

        with h5py.File(hdf5_bytes, "w") as f:
            for path, node in self.nodes.items():
                # 1.) HDF5 file has no root node ==> can be skipped
                if isinstance(node, RootNode):
                    continue

                # 2.) If the node is a group, create a group in the HDF5 file and add attributes
                elif isinstance(node, GroupNode):
                    group = f.create_group(path)
                    self._add_attributes(group, node.attributes)

                # 3.) If the node is a dataset, create a dataset in the HDF5 file and add attributes
                elif isinstance(node, DataNode):
                    # Get the data from the node and convert to NumPy array ==> h5py does not support PyTorch tensors
                    # Force=True in case the tensor is on the GPU
                    array = node.data.numpy(force=True)

                    # Create the dataset in the HDF5 file and add attributes
                    dataset = f.create_dataset(path, data=array, compression="gzip", compression_opts=9)
                    self._add_attributes(dataset, node.attributes)

                # 4.) If the node is neither a group nor a dataset, raise an error
                else:
                    raise TypeError(f"Node type '{type(node)}' is not supported for serialization.")

        # Reset the BytesIO object to the beginning and return it
        hdf5_bytes.seek(0)
        return hdf5_bytes

    def _add_attributes(self, h5obj: h5py.Dataset | h5py.Group, attributes: ItemsView[str, AttributeTypes]):
        """Adds attributes to an HDF5 object (group or dataset).

        Parameters:
            h5obj (h5py.Group or h5py.Dataset): The HDF5 object to add attributes to.
            attributes (Dict[str, str | int | float | list | tuple | dict]): The attributes to add.
        """
        for key, value in attributes:
            # Convert lists and dictionaries to JSON strings ==> HDF5 does not support these types natively
            if isinstance(value, (list, dict)):
                value = json.dumps(value)

            # Assign attribute to HDF5 object
            h5obj.attrs[key] = value

    def save_to_hdf5(self, path: str):
        """Saves the serialized DataContainer to an HDF5 file at the specified path.

        Parameters:
            path (str): The path where the HDF5 file should be saved.
        """
        with open(path, "wb") as file:
            file.write(self.serialize_to_hdf5().getvalue())


class DeserializationOps(GroupOps, DatasetOps, AttributeOps):
    def deserialize(self, hdf5_bytes: io.BytesIO):
        """Deserializes an HDF5 file into the DataContainer instance.

        Parameters:
            hdf5_bytes (io.BytesIO): The serialized HDF5 data as a BytesIO object.
        """
        # Check if the BytesIO object is empty
        if hdf5_bytes.getbuffer().nbytes == 0:
            raise ValueError("The given BytesIO object is empty.")

        # Check if the BytesIO object is a HDF5 file
        try:
            h5py.File(hdf5_bytes)
        except OSError:
            raise ValueError("The given BytesIO object does not contain a valid HDF5 file.")

        # Reset the position of the BytesIO object to the start in case the pointer was moved by the h5py.File function
        hdf5_bytes.seek(0)

        # Check if a root node exists in the DataContainer ==> if not, add it
        if not self._is_rootnode("/"):
            self.nodes["/"] = RootNode()

        # Open the HDF5 file in read mode
        with h5py.File(hdf5_bytes, "r") as f:
            # Iterate over all items in the HDF5 file
            for item_name, item in f.items():
                if isinstance(item, h5py.Group):
                    self._process_group("/", item_name, item)

                elif isinstance(item, h5py.Dataset):
                    self._process_dataset("/", item_name, item)

                else:
                    raise TypeError(f"Item type '{type(item)}' is not supported for deserialization.")

    def _process_group(self, path: str, name: str, group: h5py.Group):
        """Processes and adds one group and its subgroups and datasets to the DataContainer.

        Parameters:
            path (str): The path to the parent where the group should be added.
            name (str): The name of the group.
            group (h5py.Group): The group to process.
        """
        # Add the group to the DataContainer
        self.add_group(path, name)

        # If the group has attributes, add them to the group in the DataContainer
        if group.attrs:
            self._process_attributes(f"{path}/{name}", dict(group.attrs))

        # Iterate over all items in the group
        for item_name, item in group.items():
            if isinstance(item, h5py.Group):
                self._process_group(f"{path}/{name}", item_name, item)

            elif isinstance(item, h5py.Dataset):
                self._process_dataset(f"{path}/{name}", item_name, item)

            else:
                raise TypeError(f"Item type '{type(item)}' is not supported for deserialization.")

    def _process_dataset(self, path: str, name: str, dataset: h5py.Dataset):
        """Processes and adds a dataset to the DataContainer.

        Parameters:
            path (str): The path to the parent group where the dataset should be added.
            name (str): The name of the dataset.
            dataset (h5py.Dataset): The dataset to process.
        """
        # Add data to dataset
        self.add_dataset(path, name, dataset[()])

        # Add dataset attributes if they exist
        if dataset.attrs:
            self._process_attributes(f"{path}/{name}", dict(dataset.attrs))

    def _process_attributes(self, path: str, attributes: dict[str, Any]):
        """Processes and adds attributes to a node in the DataContainer.

        Parameters:
            path (str): The path to the node.
            attributes (Dict[str, Any]): The attributes to add to the node.
        """
        # validate the path using the utility function
        path = validate_path(path)

        # Iterate over all attributes
        for key, value in attributes.items():
            # Convert JSON strings back to lists and dictionaries
            # Try to convert the string to a list or dictionary using the JSON format
            try:
                if isinstance(value, str):
                    value_decoded = json.loads(value)

                    # Use the JSON-decoded value only if it successfully decodes to a list or dictionary
                    if isinstance(value_decoded, (list, dict)):
                        value = value_decoded

            # If it fails, the value is not a JSON object or a string ==> keep it as it is
            except (json.JSONDecodeError, TypeError):
                pass

            # Add the attribute to the node in the DataContainer
            self.add_attribute(path, key, value)

    def load_from_hdf5(self, path: str):
        """Loads an HDF5 file from the specified path and deserializes it into the DataContainer.

        Parameters:
            path (str): The path of the HDF5 file to load.
        """
        with open(path, "rb") as file:
            hdf5_bytes = io.BytesIO(file.read())
        self.deserialize(hdf5_bytes)
