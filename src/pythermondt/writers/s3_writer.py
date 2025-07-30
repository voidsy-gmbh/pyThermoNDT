import boto3
from botocore.exceptions import ClientError
from tqdm.auto import tqdm

from ..data import DataContainer
from .base_writer import BaseWriter


class S3Writer(BaseWriter):
    def __init__(self, bucket: str, destination_folder: str, boto3_session: boto3.Session | None = None):
        """Instantiates a new HDF5Writer.

        Parameters:
            bucket (str): The name of the bucket to write to.
            destination_folder (str): The destination folder where the DataContainers should be written to.
            boto3_session (boto3.Session, optional): The boto3 session to be used for the S3 client.
                Default is a new boto3 session with the default profile.
        """
        # Use default boto3 session if none is provided
        if not boto3_session:
            boto3_session = boto3.Session()

        self.bucket = bucket
        self.destination_folder = destination_folder

        # Create a new s3 client from the give session
        self.__client = boto3_session.client("s3")

    def write(self, container: DataContainer, file_name: str):
        # Specify the destination path
        if self.destination_folder:
            path = "/".join([self.destination_folder, file_name])

        else:
            path = file_name

        # Serialize the DataContainer to a HDF5 file
        hdf5_buffer = container.serialize_to_hdf5()

        # Progress bar for uploading the file
        pbar = tqdm(
            total=hdf5_buffer.getbuffer().nbytes,
            desc=f"Uploading file: {file_name}",
            unit="B",
            unit_scale=True,  # Scale to MB
            unit_divisor=1024,  # Convert bytes to MB
            miniters=1,  # Update progress bar every 1 iteration
            leave=True,  # Set to False if you don't want the bar to persist after completion
        )

        # Callback for progress bar
        class ProgressCallback:
            def __init__(self, progress_bar: tqdm):
                self.progress_bar = progress_bar

            def __call__(self, bytes_amount):
                self.progress_bar.update(bytes_amount)

        # Try to upload the file
        try:
            self.__client.upload_fileobj(hdf5_buffer, self.bucket, path, Callback=ProgressCallback(pbar))
            pbar.close()  # Close the progress bar

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in [
                "InvalidAccessKeyId",
                "SignatureDoesNotMatch",
                "AuthFailure",
                "InvalidSecurity",
                "InvalidToken",
            ]:
                raise PermissionError("Invalid AWS credentials") from e
            elif error_code == "AccessDenied":
                raise PermissionError("Access denied. Check your AWS permissions for this resource.") from e
            elif error_code == "NoSuchBucket":
                raise FileNotFoundError(f"The bucket '{self.bucket}' does not exist") from e
            else:
                raise RuntimeError(f"Failed to upload file: {e}") from e
