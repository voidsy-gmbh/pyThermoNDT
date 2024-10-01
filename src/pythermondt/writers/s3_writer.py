import boto3
import progressbar
from botocore.exceptions import ClientError
from ..data import DataContainer
from .base_writer import BaseWriter

class S3Writer(BaseWriter):
    def __init__(self, bucket: str, destination_folder: str, boto3_session: boto3.Session = boto3.Session()):
        """ Instantiates a new HDF5Writer 

        Parameters:
            bucket (str): The name of the bucket to write to.
            destination_folder (str): The destination folder where the DataContainers should be written to.
            boto3_session (boto3.Session, optional): The boto3 session to be used for the S3 client. Default is a new boto3 session with the default profile.
        """
        self.bucket = bucket
        self.destination_folder = destination_folder

        # Create a new s3 client from the give session
        self.__client = boto3_session.client('s3')

    def write(self, container: DataContainer , file_name: str):
        # Specify the destination path
        if self.destination_folder:
            path = "/".join([self.destination_folder, file_name])

        else:
            path = file_name

        # Serialize the DataContainer to a HDF5 file
        file_obj = container.serialize_to_hdf5()

        # Progress bar
        widgets = [
            f"\033[1mUploading file: {file_name}\033[0m ",  # Bold file name
            progressbar.Percentage(),
            ' ', progressbar.Bar(marker='\033[92m■\033[0m', left='|', right='|', fill=' '),
            ' ', progressbar.DataSize(), ' of ', progressbar.DataSize(variable='max_value'),
            ' ', progressbar.FileTransferSpeed(),
            ' ', progressbar.ETA()
        ]
        bar = progressbar.ProgressBar(
            max_value=file_obj.getbuffer().nbytes, 
            widgets=widgets,
            enable_colors=False
        )

         # Callback for progress bar
        class ProgressCallback:
            def __init__(self, bar):
                self.bar = bar
                self._seen_so_far = 0

            def __call__(self, bytes_amount):
                self._seen_so_far += bytes_amount
                self.bar.update(self._seen_so_far)

        # Try to upload the file
        try:
            self.__client.upload_fileobj(file_obj, self.bucket, path, Callback=ProgressCallback(bar))
            bar.finish()

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch', 'AuthFailure', 'InvalidSecurity', 'InvalidToken']:
                raise PermissionError("Invalid AWS credentials") from e
            elif error_code == 'AccessDenied':
                raise PermissionError("Access denied. Check your AWS permissions for this resource.") from e
            elif error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"The bucket '{self.bucket}' does not exist") from e
            else:
                raise RuntimeError(f"Failed to upload file: {e}") from e