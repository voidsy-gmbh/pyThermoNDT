from azure.core.credentials import TokenCredential

from ..data import DataContainer
from ..data.datacontainer.serialization_ops import CompressionType
from ..io import AzureBlobBackend, IOPathWrapper
from .base_writer import BaseWriter


class AzureBlobWriter(BaseWriter):
    def __init__(
        self,
        account_url: str,
        container_name: str,
        prefix: str,
        connection_string: str | None = None,
        credential: TokenCredential | None = None,
    ):
        """Initialize writer for Azure Blob Storage.

        Uses DefaultAzureCredential by default (managed identity, az login, env vars, etc.).
        Explicit connection_string or credential override the default behavior.

        Args:
            account_url (str): Storage account URL (https://<account>.blob.core.windows.net)
            container_name (str): Azure blob container name
            prefix (str): Prefix (folder path) within container
            connection_string (str | None): Optional connection string (overrides default auth)
            credential (TokenCredential | None): Optional Azure TokenCredential (overrides default auth)
        """
        super().__init__()
        self.__account_url = account_url
        self.__container_name = container_name
        self.__prefix = prefix
        self.__connection_string = connection_string
        self.__credential = credential

    def _create_backend(self) -> AzureBlobBackend:
        """Create a new AzureBlobBackend instance."""
        # pylint: disable=duplicate-code
        return AzureBlobBackend(
            account_url=self.__account_url,
            container_name=self.__container_name,
            prefix=self.__prefix,
            connection_string=self.__connection_string,
            credential=self.__credential,
        )
        # pylint: enable=duplicate-code

    def write(
        self,
        container: DataContainer,
        file_name: str,
        compression: CompressionType = "lzf",
        compression_opts: int | None = 4,
    ):
        """Write DataContainer to Azure Blob Storage.

        Args:
            container (DataContainer): The data container to write
            file_name (str): Name of the file to write (will append .hdf5 if missing)
            compression (CompressionType): Compression algorithm to use. Default: "lzf"
            compression_opts (int | None): Compression level (algorithm-dependent). Default: 4
        """
        # Append file extension if not present
        if not file_name.endswith(".hdf5"):
            file_name += ".hdf5"

        # Write the DataContainer to the file
        data = container.serialize_to_hdf5(compression, compression_opts)
        self.backend.write_file(IOPathWrapper(data), file_name)
