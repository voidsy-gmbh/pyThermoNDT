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
            logger.debug("Client initialized using account URL and credential.")

        self.__container_name = container_name
        self.__prefix = prefix.rstrip("/") if prefix else ""

        logger.info(f"Azure backend: container={container_name}, prefix={prefix or '(root)'}")

    @property
    def remote_source(self) -> bool:
        return True

    @property
    def container_name(self) -> str:
        return self.__container_name

    @property
    def prefix(self) -> str:
        return self.__prefix

    def read_file(self, file_path: str) -> IOPathWrapper:
        """Read file from Azure Blob Storage."""
        blob_name = self._parse_path(file_path)

        try:
            blob_client = self.__client.get_blob_client(container=self.__container_name, blob=blob_name)

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
            logger.error(e)
            raise FileNotFoundError(f"Blob not found: {file_path}") from e

    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        """Write file to Azure Blob Storage."""
        blob_name = self._parse_path(file_path)

        # Reset and get size
        data.file_obj.seek(0)
        file_size = data.file_obj.getbuffer().nbytes

        try:
            blob_client = self.__client.get_blob_client(container=self.__container_name, blob=blob_name)

            # Wrap file object to track read progress
            data.file_obj.seek(0)
            with tqdm.wrapattr(
                data.file_obj, "read", desc=f"Uploading {blob_name}", **get_tqdm_default_kwargs(file_size=file_size)
            ) as wrapped_file:
                blob_client.upload_blob(cast(IO[bytes], wrapped_file), overwrite=True, max_concurrency=4)

        except Exception as e:
            logger.error(e)
            raise RuntimeError(f"Failed to upload blob: {e}") from e

    def exists(self, file_path: str) -> bool:
        """Check if blob exists."""
        blob_name = self._parse_path(file_path)

        try:
            blob_client = self.__client.get_blob_client(container=self.__container_name, blob=blob_name)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            # Other exceptions (auth, network) should be re-raised
            logger.exception("Error checking blob existence: %s", e)
            raise

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

            # Match S3's pattern: use azure:// scheme
            blob_uri = f"azure://{self.__container_name}/{blob.name}"
            blobs.append(blob_uri)

        if extensions:
            blobs = [b for b in blobs if any(b.lower().endswith(ext.lower()) for ext in extensions)]

        blobs.sort()

        if num_files is not None:
            blobs = blobs[:num_files]

        return blobs

    def get_file_size(self, file_path: str) -> int:
        """Get blob size in bytes."""
        blob_name = self._parse_path(file_path)

        blob_client = self.__client.get_blob_client(container=self.__container_name, blob=blob_name)

        properties = blob_client.get_blob_properties()
        return properties.size

    def download_file(self, source_path: str, destination_path: str) -> None:
        """Download blob to local filesystem."""
        blob_name = self._parse_path(source_path)

        blob_client = self.__client.get_blob_client(container=self.__container_name, blob=blob_name)

        # Get file size for progress
        properties = blob_client.get_blob_properties()
        file_size = properties.size

        with open(destination_path, "wb") as local_file:
            with TqdmCallback(total=file_size, desc=f"Downloading {blob_name}") as progress:
                stream = blob_client.download_blob()
                for chunk in stream.chunks():
                    local_file.write(chunk)
                    progress.callback(len(chunk))

    def _parse_path(self, path: str) -> str:
        """Parse path into blob name."""
        # Handle azure:// URIs (new format)
        if path.startswith("azure://"):
            path = path[8:]  # Remove "azure://"
            parts = path.split("/", 1)
            # parts[0] is container, parts[1] is blob path
            if len(parts) > 1:
                return parts[1]
            return ""

        # Handle relative paths
        if self.__prefix:
            path = path.lstrip("/")
            return f"{self.__prefix}/{path}"

        return path
