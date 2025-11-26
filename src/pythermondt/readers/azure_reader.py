from azure.core.credentials import TokenCredential

from ..io import AzureBlobBackend, BaseParser
from .base_reader import BaseReader


class AzureBlobReader(BaseReader):
    def __init__(
        self,
        account_url: str,
        container_name: str,
        prefix: str,
        connection_string: str | None = None,
        credential: TokenCredential | None = None,
        num_files: int | None = None,
        download_files: bool = False,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
    ):
        """Initialize reader for Azure Blob Storage.

        Uses DefaultAzureCredential by default (managed identity, az login, env vars, etc.).
        Explicit connection_string or credential override the default behavior.

        Args:
            account_url (str): Storage account URL (https://<account>.blob.core.windows.net)
            container_name (str): Azure blob container name
            prefix (str): Prefix (folder path) within container
            connection_string (str | None): Optional connection string (overrides default auth)
            credential (TokenCredential | None): Optional Azure TokenCredential (overrides default auth)
            num_files (int | None): Maximum number of files to read. If None, reads all files.
            download_files (bool): If True, downloads and caches files locally during operations.
                If False, files are accessed on-demand without local caching. Default: False.
            cache_files (bool): If True, caches the file list in memory. If False, file list is
                refreshed on each access. Default: True.
            parser (type[BaseParser] | None): Parser class for reading files. If None, auto-selects
                based on file extension. Default: None.
        """
        super().__init__(num_files, download_files, cache_files, parser)
        self.__container_name = container_name
        self.__prefix = prefix
        self.__connection_string = connection_string
        self.__account_url = account_url
        self.__credential = credential

    def _create_backend(self) -> AzureBlobBackend:
        """Create a new AzureBlobBackend instance.

        Called to create or recreate the backend when needed or after unpickling.
        """
        return AzureBlobBackend(
            account_url=self.__account_url,
            container_name=self.__container_name,
            prefix=self.__prefix,
            connection_string=self.__connection_string,
            credential=self.__credential,
        )

    def _get_reader_params(self) -> str:
        """Get string representation of reader parameters."""
        auth_method = (
            "connection_string"
            if self.__connection_string
            else "credential"
            if self.__credential
            else "DefaultAzureCredential"
        )
        return f"container={self.__container_name}, prefix={self.__prefix}, auth={auth_method}"
