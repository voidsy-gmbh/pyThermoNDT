import boto3

from ..io import BaseParser, S3Backend
from .base_reader import BaseReader


class S3Reader(BaseReader):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        region_name: str | None = None,
        profile_name: str | None = None,
        num_files: int | None = None,
        download_remote_files: bool = False,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
        boto3_session: boto3.Session | None = None,
    ):
        """Initialize an instance of the S3Reader class.

        This class is used to read data from an S3 bucket, using the the boto3 SDK. For using this class, the user must
        configure an authentication method for boto3, according to the documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

        Parameters:
            source (str): The source of the data. This must be a valid S3 path, specified in the format: s3://bucket-name/Prefix/[.ext].
                All files that start with the provided prefix will be read. Specify the file extension if you want to
                autoselect a parser based on the file extension.
            cache_files (bool, optional): If True, all the files are downloaded first and the paths are cached in
                memory. This means the reader only checks for new files once, so changes to the file sources will not be
                noticed at runtime. Default is False, to prevent disk space issues.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data.
                If not specified, the parser will be auto selected based on the file extension. Default is None.
            num_files (int, optional): Limit the number of files that the reader can read.
                If None, the reader reads all files. Default is None.
            boto3_session (boto3.Session, optional): The boto3 session to be used for the S3 client.
                Default is a new boto3 session with the default profile.
        """
        # Use default boto3 session if none is provided
        if not boto3_session:
            boto3_session = boto3.Session()

        # Create a new s3 client from the give session
        self.__client = boto3_session.client("s3")

        # Validate the source path
        if not re.match(r"^s3:\/\/[a-z0-9][a-z0-9.-]{1,61}[a-z0-9](?:\/[\w\s.-]+)*$", source):
            raise ValueError(
                "The source must be a valid S3 path, specified in the format: s3://bucket-name/Prefix/[.ext]"
            )

        # Extract the bucket and prefix from the source path
        ext = re.findall(r"\.[a-zA-Z0-9]+$", source)
        bucket = source.split("/")[2]
        prefix = "/".join(source.split("/")[3:]) if not ext else "/".join(source.split("/")[3:-1])

        # validate that the bucket exists
        if bucket not in [response["Name"] for response in self.__client.list_buckets()["Buckets"]]:
            raise ValueError(f"The specified bucket: {bucket} does not exist for the current session: {boto3_session}.")

        # Write the bucket and prefix to the private attributes
    ):
        """Initialize an instance of the S3Reader class.

        Uses the S3Backend to read files from an specified S3 bucket. Before using the S3Reader, make sure to set up an
        authentication method for AWS, according to the documentation:
        https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

        Parameters:
            bucket (str): The name of the S3 bucket.
            prefix (str): The prefix (path) within the S3 bucket.
            region_name (str, optional): The AWS region name, e.g. eu-central-1 used when creating new connections.
                Default is None, which uses the default region from the AWS configuration.
            profile_name (str, optional): The AWS profile name to use for authentication. Default is None, which uses
                the default profile from the AWS configuration.
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
            download_remote_files (bool, optional): Wether to download remote files to local storage. Recommended to set
                to True if frequent access to the same files is needed. Default is False to avoid unnecessary downloads.
            cache_files (bool, optional): Wether to cache the files list in memory. If set to False, changes to the
                detected files will be reflected at runtime. Default is True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
        """
        # Initialize baseclass with parser
        super().__init__(num_files, download_remote_files, cache_files, parser)

        # Maintain state for what is needed to create the backend
        self.__bucket = bucket
        self.__prefix = prefix
        self.__region_name = region_name
        self.__profile_name = profile_name

    def _create_backend(self) -> S3Backend:
        """Create a new S3Backend instance.

        This method is called to create or recreate the backend when needed or after unpickling.
        """
        session = boto3.Session(
            region_name=self.__region_name,
            profile_name=self.__profile_name,
        )
        return S3Backend(self.__bucket, self.__prefix, session)

    def _get_reader_params(self) -> str:
        """Get a string representation of the reader parameters used to create the backend."""
        return (
            f"bucket={self.__bucket}, prefix={self.__prefix}, "
            f"region_name={self.__region_name}, profile_name={self.__profile_name}"
        )
