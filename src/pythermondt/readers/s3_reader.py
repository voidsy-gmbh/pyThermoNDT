import io
import re
import boto3
from typing import Type, List, Optional
from .base_reader import BaseReader
from .parsers import BaseParser

class S4Reader(BaseReader):
    def __init__(self, source: str, cache_files: bool = False, parser: Optional[Type[BaseParser]] = None, num_files: Optional[int] = None, boto3_session: boto3.Session = boto3.Session()):
        """ Initialize an instance of the S3Reader class.

        This class is used to read data from an S3 bucket, using the the boto3 SDK. For using this class, the user must cofigure an authentication method
        for boto3, according to the documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

        Parameters:
            source (str): The source of the data. This must be a valid S3 path, specified in the format: s3://bucket-name/Prefix/[.ext]. All files that start with the provided prefix will be read. Specifiy the file extension if you want to autoselect a parser based on the file extension.
            cache_files (bool, optional): If True, all the files are downloaded first and the paths are cached in memory. This means the reader only checks for new files once, so changes to the file sources will not be noticed at runtime. Default is False, to prevent disk space issues.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified, the parser will be auto selected based on the file extension. Default is None.
            num_files (int, optional): Limit the number of files that the reader can read. If None, the reader reads all files. Default is None.
            boto3_session (boto3.Session, optional): The boto3 session to be used for the S3 client. Default is a new boto3 session with the default profile.
        """

        # Create a new s3 client from the give session
        self.__client = boto3_session.client('s3')

        # Validate the source path
        if not re.match(r"^s3:\/\/[a-z0-9][a-z0-9.-]{1,61}[a-z0-9](?:\/[\w\s.-]+)*$", source):
            raise ValueError("The source must be a valid S3 path, specified in the format: s3://bucket-name/Prefix/[.ext]")
        
        # Extract the bucket and prefix from the source path
        ext = re.findall(r'\.[a-zA-Z0-9]+$', source) 
        bucket = source.split('/')[2]
        prefix = '/'.join(source.split('/')[3:]) if not ext else '/'.join(source.split('/')[3:-1])

        # validate that the bucket exists
        if not bucket in [response['Name'] for response in self.__client.list_buckets()['Buckets']]:
            raise ValueError(f"The specified bucket: {bucket} does not exist for the current session: {boto3_session}.")

        # Write the bucket and prefix to the private attributes
        self.__bucket = bucket
        self.__prefix = prefix
        
        # Call the constructor of the BaseReader class
        super().__init__(source, cache_files, parser, num_files)

    @property
    def remote_source(self) -> bool:
        return True

    def _read_file(self, path: str) -> io.BytesIO:
        # Extract the bucket and the key from the path
        bucket = path.split('/')[2]
        key = '/'.join(path.split('/')[3:])

        response = self.__client.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(response['Body'].read())

    def _get_file_list(self, num_files: Optional[int] = None) -> List[str]:
        # Create a paginator for the list_objects_v2 method ==> The amout of objects to get with list_objects_v2 is limited to 1000 
        # ==> In that case the requests are split into multiple pages which the paginator can iterate over
        paginator = self.__client.get_paginator('list_objects_v2')

        # Iterate over all pages and get the content of the objects
        files: List[str] = []
        for page in paginator.paginate(Bucket=self.__bucket, Prefix=self.__prefix):
            if page.get('Contents') is not None:
                files.extend([content['Key'] for content in page.get('Contents')])

        # Filter the files based on the file extensions and append the prefix "s3://bucket-name/" to the file paths
        files = [f"s3://{self.__bucket}/" + file for file in files if file.endswith(self.file_extensions)]

        # Limit to the first num_files if specified
        return files[:num_files] if num_files else files
    
    def _close(self):
        # Close the underlying endpoint connections of the client
        self.__client.close()