import logging

from ..data.datacontainer import DataContainer
from ..data.datacontainer.utils import is_datanode
from .base import ThermoTransform

logger = logging.getLogger(__name__)


class DeviceTransfer(ThermoTransform):
    """A transform that moves all the datasets inside a datacontainer to a specific device."""

    def __init__(self, datasets: list[str] | None = None, device: str = "cpu"):
        """Initialize the DeviceTransfer transform with specified datasets.

        Args:
            datasets (list[str]): List of dataset identifiers to be used in the transform.
            device (str): The target device to which datasets will be moved.
        """
        super().__init__()
        self.datasets = datasets
        self.device = device

    def forward(self, container: DataContainer) -> DataContainer:
        for path in container.get_all_dataset_paths() if self.datasets is None else self.datasets:
            node = container.nodes[path]
            if is_datanode(node):
                node.data = node.data.to(self.device)

        logger.debug(f"Moved datasets {self.datasets if self.datasets else 'all'} to device {self.device}")
        return container
