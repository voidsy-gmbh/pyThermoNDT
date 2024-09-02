import boto3
import os
from botocore.exceptions import ClientError
from ..data import DataContainer
from .base_writer import BaseWriter

class AWSWriter(BaseWriter):
    def __init__(self, bucket: str, destination_folder: str):
        """ Instantiates a new HDF5Writer 

        Parameters:
            bucket (str): The name of the bucket to write to.
            destination_folder (str): The destination folder where the DataContainers should be written to.
        """
        self.bucket = bucket
        self.destination_folder = destination_folder

    def write(self, container: DataContainer , file_name: str):
        # Initialize boto3 client
        s3_client = boto3.client('s3')

        if self.destination_folder:
            path = "/".join([self.destination_folder, file_name])

        else:
            path = file_name

        # Try to upload the file
        try:
            s3_client.upload_fileobj(container.serialize_to_hdf5(), self.bucket, path)
            print("Upload successful")
        except ClientError as e:
            print(e)