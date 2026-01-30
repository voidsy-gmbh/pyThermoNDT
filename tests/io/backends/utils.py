from dataclasses import dataclass
from io import BytesIO

from pythermondt.io import BaseBackend


@dataclass
class TestConfig:
    """Configuration for backend testing."""

    backend_cls: type[BaseBackend]
    scheme: str
    is_remote: bool


class MockAzureBlob:
    """Mock Azure Blob Storage for testing."""

    def __init__(self):
        self.storage: dict[str, dict[str, bytes]] = {}

    def create_container(self, container_name: str):
        if container_name not in self.storage:
            self.storage[container_name] = {}

    def upload_blob(self, container: str, blob_name: str, data: bytes | BytesIO):
        if container not in self.storage:
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Container '{container}' not found")

        # Handle both bytes and file-like objects
        if isinstance(data, BytesIO):
            data.seek(0)
            content = data.read()
        else:
            content = data

        self.storage[container][blob_name] = content

    def download_blob(self, container: str, blob_name: str) -> bytes:
        if not self.blob_exists(container, blob_name):
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Blob '{blob_name}' not found")
        return self.storage[container][blob_name]

    def blob_exists(self, container: str, blob_name: str) -> bool:
        return container in self.storage and blob_name in self.storage[container]

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        if container not in self.storage:
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Container '{container}' not found")

        blobs = list(self.storage[container].keys())
        if prefix:
            blobs = [b for b in blobs if b.startswith(prefix)]
        return blobs

    def get_blob_size(self, container: str, blob_name: str) -> int:
        if not self.blob_exists(container, blob_name):
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Blob '{blob_name}' not found")
        return len(self.storage[container][blob_name])
