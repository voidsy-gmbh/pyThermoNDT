import logging
from collections.abc import Generator
from io import BytesIO
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ..utils import IOPathWrapper
from .base_backend import BaseBackend, FileInfo
from .options import S3ClientOptions
from .progress import TqdmCallback

logger = logging.getLogger(__name__)


class S3Backend(BaseBackend):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        session: boto3.Session | None = None,
        client_options: S3ClientOptions | None = None,
    ) -> None:
        # Use default boto3 session if none is provided
        if not session:
            logger.debug("No boto3 session provided, creating default session.")
            session = boto3.Session()

        # Create a new s3 client from the given session
        config = Config(**client_options.as_kwargs()) if client_options else None
        self.__client = session.client("s3", config=config)

        # Write the bucket and prefix to the private attributes
        self.__bucket = bucket
        self.__prefix = prefix
        logger.debug(
            "S3Backend initialized: bucket=%s, prefix=%s, client_options=%s",
            bucket,
            prefix,
            client_options or "default",
        )

    @property
    def remote_source(self) -> bool:
        # This backend is always remote
        return True

    @property
    def scheme(self) -> str:
        return "s3"

    @property
    def bucket(self) -> str:
        return self.__bucket

    @property
    def prefix(self) -> str:
        return self.__prefix

    def read_file(self, file_path: str) -> IOPathWrapper:
        """Read a file from S3.

        Args:
            file_path (str): Path to file, either full S3 URI or key within bucket

        Returns:
            IOPathWrapper: File contents

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        bucket, key = self._parse_input(file_path)
        data = BytesIO()
        with TqdmCallback(total=self.get_file_size(file_path), desc=f"Downloading {key}") as pbar:
            self.__client.download_fileobj(bucket, key, data, Callback=pbar.callback)
        return IOPathWrapper(data)

    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        """Write file to S3.

        Args:
            data (IOPathWrapper): File data to write
            file_path (str): Destination path
        """
        bucket, key = self._parse_input(file_path)

        # Reset file object position
        data.file_obj.seek(0)

        # Upload to S3 (Always show progress)
        try:
            with TqdmCallback(total=data.file_obj.getbuffer().nbytes, desc=f"Uploading {key}") as pbar:
                self.__client.upload_fileobj(data.file_obj, bucket, key, Callback=pbar.callback)
        except ClientError as e:
            raise RuntimeError(f"Failed to upload file to S3: {e}") from e

    def exists(self, file_path: str) -> bool:
        """Check if a file exists in S3.

        Args:
            file_path (str): Path to check

        Returns:
            bool: True if file exists
        """
        bucket, key = self._parse_input(file_path)

        try:
            self.__client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if self._is_not_found_error(e):
                return False
            raise

    def close(self) -> None:
        """Close connections.

        For S3, we need to close the underlying boto3 client.
        """
        self.__client.close()

    def _iter_objects(self, prefix: str) -> Generator[dict]:
        """Yield each non-directory object dict under the configured prefix.

        Directory markers (keys ending with ``/``) are skipped.

        Args:
            prefix (str): Prefix to filter objects

        Yields:
            dict: Each object's metadata from S3
        """
        paginator = self.__client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if not obj["Key"].endswith("/"):
                        yield obj

    def get_file_list(self) -> list[str]:
        """Return all objects under the configured prefix as unsorted S3 URIs."""
        return [self._to_url(self.bucket, obj["Key"]) for obj in self._iter_objects(self.prefix)]

    def get_file_list_with_metadata(self) -> list[FileInfo]:
        """Return all objects under the configured prefix as ``FileInfo`` entries (unsorted).

        Metadata (``LastModified``, ``Size``, ``ETag``) is gathered from the
        ``list_objects_v2`` response — no extra HEAD requests.
        """
        return [
            FileInfo(
                path=self._to_url(self.bucket, obj["Key"]),
                last_modified=obj["LastModified"],
                size=obj["Size"],
                file_identity=obj.get("ETag"),
            )
            for obj in self._iter_objects(self.prefix)
        ]

    def get_file_size(self, file_path: str) -> int:
        """Return the size of the file on s3 bucket in bytes."""
        bucket, key = self._parse_input(file_path)

        try:
            response = self.__client.head_object(Bucket=bucket, Key=key)
            return response["ContentLength"]
        except ClientError as e:
            if self._is_not_found_error(e):
                raise FileNotFoundError(f"File not found: {file_path}") from e
            raise

    def get_file_identity(self, file_path: str) -> str:
        """Return object identity (ETag) for a file on S3.

        Args:
            file_path (str): Path to file on S3.

        Returns:
            str: ETag value.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        bucket, key = self._parse_input(file_path)

        try:
            response = self.__client.head_object(Bucket=bucket, Key=key)
            etag = response.get("ETag")
            if etag is None:
                raise RuntimeError(f"ETag unavailable for file: {file_path}")
            return etag
        except ClientError as e:
            if self._is_not_found_error(e):
                raise FileNotFoundError(f"File not found: {file_path}") from e
            raise

    def download_file(self, source_path: str, destination_path: str) -> None:
        """Download a file from S3 to local filesystem.

        Args:
            source_path (str): Source S3 path
            destination_path (str): Destination local path
        """
        bucket, key = self._parse_input(source_path)

        # Download the file
        with TqdmCallback(total=self.get_file_size(source_path), desc=f"Downloading {key}") as progress:
            self.__client.download_file(bucket, key, destination_path, Callback=progress.callback)

    def _parse_input(self, file_path: str) -> tuple[str, str]:
        """Convert S3 URI to (bucket, key) tuple.

        Args:
            file_path: Either "s3://bucket/key" or just "key"

        Returns:
            tuple[str, str]: (bucket, key)
        """
        parsed = urlparse(file_path)
        if parsed.scheme == "s3":
            # s3://bucket/key/path -> bucket="bucket", key="key/path"
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")  # Remove leading slash
            return bucket, key

        # Not a URI - treat as absolute key within default bucket
        # No prefix prepending - user must provide full key
        return self.bucket, file_path

    def _to_url(self, bucket: str, key: str) -> str:
        """Convert (bucket, key) to S3 URI.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            str: S3 URI like "s3://bucket/key"
        """
        return f"s3://{bucket}/{key}"

    def _is_not_found_error(self, e: ClientError) -> bool:
        """Check if ClientError indicates file not found.

        AWS S3 returns different error codes for "not found":
        - '404' - from head_object() operation
        - 'NoSuchKey' - from get_object() and other operations
        - 'NoSuchBucket' - bucket doesn't exist
        - '403' - can indicate missing file if user lacks s3:ListBucket permission

        Args:
            e: ClientError from boto3

        Returns:
            bool: True if this is a not-found error
        """
        error_code = e.response.get("Error", {}).get("Code", "")
        return error_code in ("404", "403", "NoSuchKey", "NoSuchBucket")
