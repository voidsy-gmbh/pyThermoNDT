from dataclasses import dataclass

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
        self.storage = {}  # {container: {blob_name: bytes}}

    def create_container(self, container_name: str):
        if container_name not in self.storage:
            self.storage[container_name] = {}

    def upload_blob(self, container: str, blob_name: str, data: bytes):
        if container not in self.storage:
            self.storage[container] = {}
        self.storage[container][blob_name] = data

    def download_blob(self, container: str, blob_name: str) -> bytes:
        return self.storage[container][blob_name]

    def blob_exists(self, container: str, blob_name: str) -> bool:
        return container in self.storage and blob_name in self.storage[container]

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        if container not in self.storage:
            return []
        blobs = self.storage[container].keys()
        if prefix:
            blobs = [b for b in blobs if b.startswith(prefix)]
        return list(blobs)

    def get_blob_size(self, container: str, blob_name: str) -> int:
        return len(self.storage[container][blob_name])
