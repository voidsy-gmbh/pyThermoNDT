from io import BytesIO

from ._comparison import _iter_node_differences
from .attribute_ops import AttributeOps
from .dataset_ops import DatasetOps
from .group_ops import GroupOps
from .node import RootNode
from .serialization_ops import DeserializationOps, SerializationOps
from .visualization_ops import VisualizationOps


class DataContainer(SerializationOps, DeserializationOps, VisualizationOps, GroupOps, DatasetOps, AttributeOps):
    """Manages and serializes data into HDF5 format.

    This class manages data in a hierarchical structure, similar to a HDF5 File. It provides methods to add
    groups, datasets and attributes to the data structure. The data structure can be serialized to a HDF5 file and
    deserialized from a HDF5 file using save_to_hdf5 and load_from_hdf5 methods respectively. It also provides methods
    for visualization of the data structure.
    """

    def __init__(self, hdf5_data: BytesIO | None = None):
        """Initializes a DataContainer instance.

        By default, initializes an empty DataContainer.
        If a serialized HDF5 file is provided, the DataContainer is initialized with the data from the BytesIO object.

        Args:
            hdf5_data (BytesIO | None): The HDF5 file to deserialize. Defaults to None.
        """
        super().__init__()

        # Add the root node to the data structure
        self.nodes["/"] = RootNode()

        # If provided, initialize from a serialized HDF5 file.
        if hdf5_data:
            self.deserialize(hdf5_data)

    # Overwrite the __str__ method to provide a string representation of the DataContainer
    def __str__(self):
        returnstring = ""
        for path, node in self.nodes.items():
            returnstring = returnstring + f"{path}: ({node.name}: {node.type})" + "\n"

        return returnstring

    def __eq__(self, other: object) -> bool:
        """Compare two DataContainers for equality.

        Uses the default comparison rules of ``containers_equal``. See its docstring for details.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: True if the two DataContainers are equal, False otherwise.
        """
        if not isinstance(other, DataContainer):
            return False

        return next(_iter_node_differences(self.nodes, other.nodes), None) is None
