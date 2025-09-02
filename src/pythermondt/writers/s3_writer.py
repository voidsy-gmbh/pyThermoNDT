import boto3

from ..data import DataContainer
from ..io import IOPathWrapper, S3Backend
from .base_writer import BaseWriter


class S3Writer(BaseWriter):
    def __init__(self, bucket: str, prefix: str, region_name: str | None = None, profile_name: str | None = None):
        """Instantiates a new instance of the S3Writer class.

        Args:
            bucket (str): The name of the bucket to write to.
            prefix (str): The prefix (folder path) within the bucket to write to.
            region_name (str | None, optional): The AWS region to use. Defaults to None.
            profile_name (str | None, optional): The AWS profile to use. Defaults to None.
                Default is a new boto3 session with the default profile.
        """
        super().__init__()

        # Use default boto3 session if none is provided
        self.__bucket = bucket
        self.__prefix = prefix
        self.__region_name = region_name
        self.__profile_name = profile_name

    def _create_backend(self) -> S3Backend:
        # pylint: disable=duplicate-code
        session = boto3.Session(region_name=self.__region_name, profile_name=self.__profile_name)
        return S3Backend(self.__bucket, self.__prefix, session)

    def write(self, container: DataContainer, file_name: str):
        # Append file extension if not present
        if not file_name.endswith(".hdf5"):
            file_name += ".hdf5"

        # Write the DataContainer to the file
        data = container.serialize_to_hdf5()
        self.backend.write_file(IOPathWrapper(data), file_name)
