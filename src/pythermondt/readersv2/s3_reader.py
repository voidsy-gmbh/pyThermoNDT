import io
import re
import boto3
from typing import Type, List
from .base_reader import BaseReader
from .parsers import BaseParser

class S3Reader(BaseReader):
    def __init__(self, parser: Type[BaseParser], source: str, cache_path: bool = True, download_files: bool = False, boto3_session: boto3.Session = boto3.Session()):
        """ Initialize an instance of the S3Reader class.

        This class is used to read data from an S3 bucket, using the the boto3 SDK. For using this class, the user must cofigure an authentication method
        for boto3, according to the documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

        Parameters:
            parser (BaseParser): The parser to be used for parsing the data.
            source (str): The source of the data. This must be a valid S3 path, specified in the format: s3://bucket-name/Prefix. All files that start with the provided prefix will be read.
            cache_path (bool, optional): If True, all the file paths are cached in memory. This means the reader only checks for new files once, so changes to the file sources will not be noticed at runtime. Default is True.
            download_files (bool, optional): If True, the reader will download the files to the current working directory on the first read. Subsequent reads will be faster by using the local files. Default is False to prevent disk space issues.
            boto3_session (boto3.Session, optional): The boto3 session to be used for the S3 client. Default is a new boto3 session with the default profile.
        """

        # Create a new s3 client from the give session
        self.__client = boto3_session.client('s3')

        # Validate the source path
        if not re.match(r"^s3:\/\/[a-z0-9][a-z0-9.-]{1,61}[a-z0-9](?:\/[\w.-]+)*$", source):
            raise ValueError("The source must be a valid S3 path, specified in the format: s3://bucket-name/path/to/file")
        
        # Extract the bucket and prefix from the source path
        bucket = source.split('/')[2]
        prefix = '/'.join(source.split('/')[3:])

        # validate that the bucket exists
        if not bucket in [response['Name'] for response in self.__client.list_buckets()['Buckets']]:
            raise ValueError(f"The specified bucket: {bucket} does not exist for the current session: {boto3_session}.")

        # Write the bucket and prefix to the private attributes
        self.__bucket = bucket
        self.__prefix = prefix
        
        # Call the constructor of the BaseReader class
        super().__init__(parser, source, cache_path, download_files)

    def _read_file(self, path: str) -> io.BytesIO:
        # Extract the bucket and the key from the path
        bucket = path.split('/')[2]
        key = '/'.join(path.split('/')[3:])

        response = self.__client.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(response['Body'].read())

    def _get_file_list(self) -> List[str]:
        # Create a paginator for the list_objects_v2 method ==> The amout of objects to get with list_objects_v2 is limited to 1000 
        # ==> In that case the requests are split into multiple pages which the paginator can iterate over
        paginator = self.__client.get_paginator('list_objects_v2')

        # Iterate over all pages and get the content of the objects
        files: List[str] = []
        for page in paginator.paginate(Bucket=self.__bucket, Prefix=self.__prefix):
            if page.get('Contents') is not None:
                files.extend([content['Key'] for content in page.get('Contents')])

        # Filter the files based on the file extensions and append the prefix "s3://bucket-name/" to the file paths
        return [f"s3://{self.__bucket}/" + file for file in files if file.endswith(self.file_extensions)]
    
    def _close(self):
        # Close the underlying endpoint connections of the client
        self.__client.close()