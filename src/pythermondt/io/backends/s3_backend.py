import boto3

from .base_backend import BaseBackend


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
