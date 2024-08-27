import io
import h5py
import json
import torch
from typing import Dict, ItemsView
from .base import BaseOps
from .node import DataNode, GroupNode, RootNode, AttributeTypes
from .utils import split_path
from .group_ops import GroupOps
from .dataset_ops import DatasetOps
from .attribute_ops import AttributeOps

class SerializationOps(BaseOps):
    def serialize_to_hdf5(self) -> io.BytesIO:
        """
        Serializes the DataContainer instance to an HDF5 file.

        Returns:
        - io.BytesIO: The serialized DataContainer instance as a BytesIO object.
        """
        hdf5_bytes = io.BytesIO()

        with h5py.File(hdf5_bytes, 'w') as f:
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
        """
        Adds attributes to an HDF5 object (group or dataset).

        Parameters:
        - h5obj (h5py.Group or h5py.Dataset): The HDF5 object to add attributes to.
        - attributes (Dict[str, str | int | float | list | dict]): The attributes to add.
        """
        for key, value in attributes:
            # Convert lists and dictionaries to JSON strings ==> HDF5 does not support these types natively
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value)

            # Assign attribute to HDF5 object
            h5obj.attrs[key] = value

    def save_to_hdf5(self, path: str):
        """
        Saves the serialized DataContainer to an HDF5 file at the specified path.

        Parameters:
        - path (str): The path where the HDF5 file should be saved.
        """
        with open(path, 'wb') as file:
            file.write(self.serialize_to_hdf5().getvalue())