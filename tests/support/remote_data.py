"""Configure Azure integration tests through TEST_* environment variables.

Set TEST_ACCOUNT_URL to the HTTPS account URL and TEST_CONTAINER_NAME to
the container; TEST_PREFIX defaults to the container root. Authentication uses the
AZURE_STORAGE_CONNECTION_STRING secret when set, otherwise DefaultAzureCredential.
Cloud tests run in CI or with REMOTE_TESTS=1; PRs without the secret skip them.
"""

from typing import Annotated

from pydantic import AnyUrl, Field, SecretStr, UrlConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AccountUrl = Annotated[AnyUrl, UrlConstraints(allowed_schemes=["https"], host_required=True)]


class CloudTestSettings(BaseSettings):
    """Cloud-test gating and credentials, loadable without an Azure location."""

    model_config = SettingsConfigDict(
        env_prefix="TEST_",
        frozen=True,
        populate_by_name=True,
        case_sensitive=False,
        env_ignore_empty=True,
        hide_input_in_errors=True,
    )

    remote_tests: bool = Field(default=False, validation_alias="REMOTE_TESTS")
    github_actions: bool = Field(default=False, validation_alias="GITHUB_ACTIONS")
    github_event_name: str = Field(default="", validation_alias="GITHUB_EVENT_NAME")
    connection_string: SecretStr | None = Field(
        default=None, validation_alias="AZURE_STORAGE_CONNECTION_STRING", repr=False, exclude=True
    )

    @field_validator("remote_tests", "github_actions", mode="before")
    @classmethod
    def strip_flag_whitespace(cls, value: object) -> object:
        """Allow whitespace around flags before Pydantic parses their boolean values."""
        return value.strip() if isinstance(value, str) else value

    @property
    def enabled(self) -> bool:
        """Whether CI or an explicit local opt-in enables cloud tests."""
        return self.github_actions or self.remote_tests


class RemoteTestConfig(CloudTestSettings):
    """Azure test-data location, validated only when cloud tests need it.

    Authentication uses the inherited connection string when set, otherwise
    DefaultAzureCredential (e.g. az login).
    """

    account_url: AccountUrl
    container_name: str = Field(min_length=1, pattern=r"\S")
    prefix: str = ""

    @field_validator("account_url")
    @classmethod
    def validate_account_endpoint(cls, value: AnyUrl) -> AnyUrl:
        """Keep credentials and container paths out of the account endpoint."""
        if (
            value.username is not None
            or value.password is not None
            or value.query is not None
            or value.fragment is not None
            or value.path not in {None, "", "/"}
        ):
            raise ValueError("Use an account URL without credentials, query, fragment, or container path.")
        return value

    @field_validator("prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        """Keep prefixes relative to the container, without changing blob-name characters."""
        return value.strip("/")


def remote_tests_enabled() -> bool:
    """Return True in GitHub Actions or after an explicit local opt-in."""
    return CloudTestSettings().enabled


def cloud_skip_reason() -> str | None:
    """Return why cloud-marked tests must skip, or None when they can run.

    Pull requests without a connection string skip instead of fail: forks cannot read
    repository secrets, and a missing secret must not break their checks.
    """
    settings = CloudTestSettings()
    if not settings.enabled:
        return "cloud tests disabled; set REMOTE_TESTS=1 to run them"
    if settings.github_event_name == "pull_request" and settings.connection_string is None:
        return "pull request without AZURE_STORAGE_CONNECTION_STRING; fork pull requests cannot read secrets"
    return None


def load_remote_config() -> RemoteTestConfig:
    """Load the test data location and Azure credentials from the environment.

    Returns:
        Connection settings for the test data container.

    Raises:
        pydantic.ValidationError: If required settings are missing or invalid.
    """
    return RemoteTestConfig()
