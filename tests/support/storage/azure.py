import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch


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

    def get_blob_etag(self, container: str, blob_name: str) -> str:
        if not self.blob_exists(container, blob_name):
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Blob '{blob_name}' not found")

        content = self.storage[container][blob_name]
        return f'"{hashlib.md5(content, usedforsecurity=False).hexdigest()}"'


@contextmanager
def mocked_azure_blob_storage() -> Iterator[MockAzureBlob]:
    """Mock Azure BlobServiceClient calls and yield in-memory storage."""
    mock_storage = MockAzureBlob()

    def make_blob_client(container: str, blob: str):
        mock_blob = MagicMock()

        def download_blob():
            data = mock_storage.download_blob(container, blob)
            mock_stream = MagicMock()
            mock_stream.chunks = lambda: iter([data])
            return mock_stream

        def upload_blob(data, overwrite=True):
            content = data.read()
            mock_storage.upload_blob(container, blob, content)

        def get_blob_properties():
            if not mock_storage.blob_exists(container, blob):
                from azure.core.exceptions import ResourceNotFoundError

                raise ResourceNotFoundError("Blob not found")
            mock_props = MagicMock()
            mock_props.size = mock_storage.get_blob_size(container, blob)
            mock_props.etag = mock_storage.get_blob_etag(container, blob)
            return mock_props

        mock_blob.download_blob = download_blob
        mock_blob.upload_blob = upload_blob
        mock_blob.get_blob_properties = get_blob_properties
        return mock_blob

    def make_container_client(container: str):
        mock_container = MagicMock()

        def list_blobs(name_starts_with=""):
            mock_blobs = []
            for name in mock_storage.list_blobs(container, name_starts_with):
                mock_blob = MagicMock()
                mock_blob.name = name
                mock_blob.last_modified = datetime(2024, 6, 1, tzinfo=timezone.utc)
                mock_blob.size = mock_storage.get_blob_size(container, name)
                mock_blob.etag = mock_storage.get_blob_etag(container, name)
                mock_blobs.append(mock_blob)
            return mock_blobs

        mock_container.list_blobs = list_blobs
        return mock_container

    with patch("pythermondt.io.backends.azure_backend.BlobServiceClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_blob_client = make_blob_client
        mock_client.get_container_client = make_container_client
        mock_client.close = MagicMock()

        mock_client_class.from_connection_string.return_value = mock_client
        mock_client_class.return_value = mock_client

        yield mock_storage
