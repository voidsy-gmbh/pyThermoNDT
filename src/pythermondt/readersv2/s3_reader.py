import io
import boto3
from typing import Type, List
from .base_reader import BaseReader
from .parsers import BaseParser


class S3Reader(BaseReader):
    def __init__(self, parser: Type[BaseParser], bucket: str, prefix: str = "", boto3_session: boto3.Session = boto3.Session()):
        """ Initialize an instance of the S3Reader class.

        This class is used to read data from an S3 bucket, using the the boto3 SDK. For using this class, the user must cofigure an authentication method
        for boto3, according to the documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

        Parameters:
            parser (BaseParser): The parser to be used for parsing the data.
            bucket (str): The name of the S3 bucket to read the data from.
            prefix (str): The prefix of the objects in the bucket to read. Limits the objects to read to those that start with the specified prefix.
            boto3_session (boto3.Session): The boto3 session to be used for the S3 client. Default is a new boto3 session with the default profile.
        """
        super().__init__(parser)
        # Create a session with the specified profile name
        session = boto3_session

        # Create a new s3 client
        self.__client = session.client('s3')

        # validate that the bucket exists
        if not bucket in [response['Name'] for response in self.__client.list_buckets()['Buckets']]:
            raise ValueError(f"The specified bucket: {bucket} does not exist for the current session: {session}.")        
        self.__prefix = prefix
        self.__bucket = bucket

    @property
    def files(self) -> List[str]:
        # Create a paginator for the list_objects_v2 method ==> The amout of objects to get with list_objects_v2 is limited to 1000 
        # ==> In that case the requests are split into multiple pages
        paginator = self.__client.get_paginator('list_objects_v2')

        # Iterate over all pages and get the content of the objects
        files: List[str] = []
        for page in paginator.paginate(Bucket=self.__bucket, Prefix=self.__prefix):
            if page.get('Contents') is not None:
                files.extend([content['Key'] for content in page.get('Contents')])

        # Filter the files based on the file extensions
        return [file for file in files if file.endswith(self.file_extensions)]

    def _read(self, path: str) -> io.BytesIO:
        response = self.__client.get_object(Bucket=self.__bucket, Key=path)
        return io.BytesIO(response['Body'].read())