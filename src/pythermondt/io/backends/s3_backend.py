from io import BytesIO

import boto3
from botocore.exceptions import ClientError

from ..utils import IOPathWrapper
from .base_backend import BaseBackend
from .progress import TqdmCallback


class S3Backend(BaseBackend):
    def __init__(self, bucket: str, prefix: str, session: boto3.Session | None = None) -> None:
        # Use default boto3 session if none is provided
        if not session:
            session = boto3.Session()

        # Create a new s3 client from the give session
        self.__client = session.client("s3")

        # Write the bucket and prefix to the private attributes
        self.__bucket = bucket
        self.__prefix = prefix

    @property
    def remote_source(self) -> bool:
        # This backend is always remote
        return True

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
        bucket, key = self._parse_path(file_path)
        try:
            data = BytesIO()
            with TqdmCallback(total=self.get_file_size(file_path), desc=f"Downloading {key}") as pbar:
                self.__client.download_fileobj(bucket, key, data, Callback=pbar.callback)
            return IOPathWrapper(data)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "NoSuchBucket"):
                raise FileNotFoundError(f"File not found: {file_path}") from e
            raise

    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        """Write file to S3.

        Args:
            data (IOPathWrapper): File data to write
            file_path (str): Destination path
        """
        bucket, key = self._parse_path(file_path)

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
        bucket, key = self._parse_path(file_path)

        try:
            self.__client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "403", "NoSuchKey"):
                return False
            raise

    def close(self) -> None:
        """Close connections.

        For S3, we need to close the underlying boto3 client.
        """
        self.__client.close()

    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        """Get list of files from S3 with optional filtering.

        Args:
            extensions (tuple[str, ...], optional): Filter by file extensions
            num_files (int, optional): Limit number of files returned

        Returns:
            list[str]: List of file paths
        """
        # List all objects with the given prefix
        paginator = self.__client.get_paginator("list_objects_v2")

        files = []
        for page in paginator.paginate(Bucket=self.__bucket, Prefix=self.__prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Skip directories (keys ending with /)
                    if key.endswith("/"):
                        continue

                    # Add full S3 path
                    files.append(f"s3://{self.__bucket}/{key}")

        # Filter by extension if provided
        if extensions:
            files = [f for f in files if any(f.lower().endswith(ext.lower()) for ext in extensions)]

        # Sort for deterministic behavior
        files.sort()

        # Limit results if specified
        if num_files is not None:
            files = files[:num_files]

        return files

    def get_file_size(self, file_path: str) -> int:
        """Return the size of the file on s3 bucket in bytes."""
        bucket, key = self._parse_path(file_path)
        response = self.__client.head_object(Bucket=bucket, Key=key)
        return response["ContentLength"]

    def download_file(self, source_path: str, destination_path: str) -> None:
        """Download a file from S3 to local filesystem.

        Args:
            source_path (str): Source S3 path
            destination_path (str): Destination local path
        """
        bucket, key = self._parse_path(source_path)

        # Download the file
        with TqdmCallback(total=self.get_file_size(source_path), desc=f"Downloading {key}") as progress:
            self.__client.download_file(bucket, key, destination_path, Callback=progress.callback)

    def _parse_path(self, path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and key.

        Handles both s3://bucket/key format and relative paths

        Args:
            path (str): Path to parse

        Returns:
            tuple[str, str]: (bucket, key)
        """
        # Handle s3:// URIs
        if path.startswith("s3://"):
            # Remove s3:// prefix
            path = path[5:]
            # Split into bucket and key
            parts = path.split("/", 1)
            if len(parts) == 1:
                # No key, just bucket
                return parts[0], ""
            return parts[0], parts[1]

        # Assume path is relative to bucket/prefix
        if self.__prefix:
            # Ensure we don't have double slashes
            path = path.removeprefix("/")
            return self.__bucket, f"{self.__prefix}/{path}"

        return self.__bucket, path
