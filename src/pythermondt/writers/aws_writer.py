import boto3
from botocore.exceptions import ClientError
from ..data import DataContainer
from .base_writer import BaseWriter

class AWSWriter(BaseWriter):
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
        if self.destination_folder:
            path = "/".join([self.destination_folder, file_name])

        else:
            path = file_name

        # Try to upload the file
        try:
            self.__client.upload_fileobj(container.serialize_to_hdf5(), self.bucket, path)
            print("Upload successful")
        except ClientError as e:
            print(e)