import logging
from io import BytesIO
from typing import IO, cast

from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from tqdm.auto import tqdm

from ..utils import IOPathWrapper
from .base_backend import BaseBackend
from .progress import TqdmCallback, get_tqdm_default_kwargs

logger = logging.getLogger(__name__)


class AzureBlobBackend(BaseBackend):
    def __init__(
        self,
        account_url: str,
        container_name: str,
        prefix: str = "",
        connection_string: str | None = None,
        credential: TokenCredential | None = None,
    ) -> None:
        """Initialize Azure Blob Storage backend.

        Uses DefaultAzureCredential by default (managed identity, az login, env vars, etc.). Providing an explicit
        connection_string or credential overrides the default behavior.

        Authentication priority:
            1. DefaultAzureCredential (default - auto-discovers credentials)
            2. connection_string (if provided overrides default behavior)
            3. credential (if provided overrides default behavior)

        Args:
            account_url (str): Storage account URL (https://<account>.blob.core.windows.net)
            container_name (str): Container name
            prefix (str, optional): Prefix within the specified container
            connection_string (str, optional): Connection string (optional, for dev/researchers)
            credential (TokenCredential, optional): Azure TokenCredential (optional, defaults to DefaultAzureCredential)
        """
        if connection_string:
            self.__client = BlobServiceClient.from_connection_string(connection_string)
            logger.debug("Client initialized using connection string.")
        else:
            if credential is None:
                credential = DefaultAzureCredential()
                logger.debug("Using DefaultAzureCredential for authentication.")
            self.__client = BlobServiceClient(account_url, credential=credential)

        self.__container_name = container_name
        self.__prefix = prefix.rstrip("/") if prefix else ""

        logger.debug(f"AzureBackend(container={container_name}, prefix={prefix}")

    @property
    def remote_source(self) -> bool:
        return True

    @property
    def scheme(self) -> str:
        return "az"

    @property
    def container_name(self) -> str:
        return self.__container_name

    @property
    def prefix(self) -> str:
        return self.__prefix

    def read_file(self, file_path: str) -> IOPathWrapper:
        """Read file from Azure Blob Storage."""
        container, blob_name = self._parse_input(file_path)

        try:
            blob_client = self.__client.get_blob_client(container=container, blob=blob_name)

            # Get blob properties for progress bar
            properties = blob_client.get_blob_properties()
            file_size = properties.size

            data = BytesIO()
            with TqdmCallback(total=file_size, desc=f"Downloading {blob_name}") as pbar:
                stream = blob_client.download_blob()
                for chunk in stream.chunks():
                    data.write(chunk)
                    pbar.callback(len(chunk))

            data.seek(0)
            return IOPathWrapper(data)

        except ResourceNotFoundError as e:
            raise FileNotFoundError(f"File not found: {file_path}") from e

    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        """Write file to Azure Blob Storage."""
        container, blob_name = self._parse_input(file_path)

        # Reset and get size
        data.file_obj.seek(0)
        file_size = data.file_obj.getbuffer().nbytes

        try:
            blob_client = self.__client.get_blob_client(container=container, blob=blob_name)

            # Wrap file object to track read progress
            data.file_obj.seek(0)
            with tqdm.wrapattr(
                data.file_obj, "read", desc=f"Uploading {blob_name}", **get_tqdm_default_kwargs(file_size=file_size)
            ) as wrapped_file:
                blob_client.upload_blob(cast(IO[bytes], wrapped_file), overwrite=True)

        except Exception as e:
            raise RuntimeError(f"Failed to upload blob: {e}") from e

    def exists(self, file_path: str) -> bool:
        """Check if blob exists."""
        container, blob_name = self._parse_input(file_path)

        try:
            blob_client = self.__client.get_blob_client(container=container, blob=blob_name)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False

    def close(self) -> None:
        """Close Azure Blob client connections."""
        self.__client.close()

    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        """Get list of blobs with optional filtering."""
        blobs = []
        container_client = self.__client.get_container_client(self.__container_name)

        blob_prefix = self.__prefix + "/" if self.__prefix else ""

        for blob in container_client.list_blobs(name_starts_with=blob_prefix):
            if blob.name.endswith("/"):
                continue

            # Add full Azure path using _to_url
            blobs.append(self._to_url(self.__container_name, blob.name))

        if extensions:
            blobs = [b for b in blobs if any(b.lower().endswith(ext.lower()) for ext in extensions)]

        blobs.sort()

        if num_files is not None:
            blobs = blobs[:num_files]

        return blobs

    def get_file_size(self, file_path: str) -> int:
        """Get blob size in bytes."""
        container, blob_name = self._parse_input(file_path)

        try:
            blob_client = self.__client.get_blob_client(container=container, blob=blob_name)

            properties = blob_client.get_blob_properties()
            return properties.size
        except ResourceNotFoundError as e:
            raise FileNotFoundError(f"File not found: {file_path}") from e

    def download_file(self, source_path: str, destination_path: str) -> None:
        """Download blob to local filesystem."""
        container, blob_name = self._parse_input(source_path)

        blob_client = self.__client.get_blob_client(container=container, blob=blob_name)

        # Get file size for progress
        properties = blob_client.get_blob_properties()
        file_size = properties.size

        with open(destination_path, "wb") as local_file:
            with TqdmCallback(total=file_size, desc=f"Downloading {blob_name}") as progress:
                stream = blob_client.download_blob()
                for chunk in stream.chunks():
                    local_file.write(chunk)
                    progress.callback(len(chunk))

    def _parse_input(self, file_path: str) -> tuple[str, str]:
        """Convert Azure URI to (container, blob_name) tuple.

        Args:
            file_path: Either "az://container/blob" or just "blob"

        Returns:
            tuple[str, str]: (container, blob_name)
        """
        if file_path.startswith("az://"):
            # az://container/blob/path -> container="container", blob_name="blob/path"
            path = file_path[5:]  # Remove "az://"
            parts = path.split("/", 1)
            container = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            return container, blob_name

        # Not a URI - treat as blob name within default container
        # Apply prefix if configured
        blob_name = file_path.lstrip("/")
        if self.__prefix:
            blob_name = f"{self.__prefix}/{blob_name}"

        return self.__container_name, blob_name

    def _to_url(self, container: str, blob_name: str) -> str:
        """Convert (container, blob_name) to Azure URI.

        Args:
            container: Azure container name
            blob_name: Blob name/key

        Returns:
            str: Azure URI like "az://container/blob"
        """
        return f"az://{container}/{blob_name}"
