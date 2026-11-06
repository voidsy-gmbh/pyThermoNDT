"""Configure Azure integration tests through TEST_* environment variables.

Set TEST_ACCOUNT_URL to the HTTPS account URL and TEST_CONTAINER_NAME to
the container; TEST_PREFIX defaults to the container root. Authentication uses
DefaultAzureCredential: az login locally, or Azure OIDC login in GitHub Actions.
Set REMOTE_TESTS=1 locally; CI enables it only after a successful trusted Azure login.
"""

from typing import Annotated

from pydantic import AnyUrl, Field, UrlConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AccountUrl = Annotated[AnyUrl, UrlConstraints(allowed_schemes=["https"], host_required=True)]


class CloudTestSettings(BaseSettings):
    """Cloud-test opt-in, loadable without an Azure location or authentication."""

    model_config = SettingsConfigDict(
        env_prefix="TEST_",
        frozen=True,
        populate_by_name=True,
        case_sensitive=False,
        env_ignore_empty=True,
        hide_input_in_errors=True,
    )

    remote_tests: bool = Field(default=False, validation_alias="REMOTE_TESTS")

    @field_validator("remote_tests", mode="before")
    @classmethod
    def strip_flag_whitespace(cls, value: object) -> object:
        """Allow whitespace around flags before Pydantic parses their boolean values."""
        return value.strip() if isinstance(value, str) else value


class RemoteTestConfig(CloudTestSettings):
    """Azure test-data location, validated only when cloud tests need it.

    Authentication uses DefaultAzureCredential (e.g. az login).
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
    """Return True after explicit opt-in, set by CI only after Azure login succeeds."""
    return CloudTestSettings().remote_tests


def cloud_skip_reason() -> str | None:
    """Return why cloud tests must skip; CI owns the decision to authenticate trusted runs."""
    if not remote_tests_enabled():
        return "cloud tests disabled; set REMOTE_TESTS=1 to run them"
    return None


def load_remote_config() -> RemoteTestConfig:
    """Load the test data location from the environment.

    Returns:
        Connection settings for the test data container.

    Raises:
        pydantic.ValidationError: If required settings are missing or invalid.
    """
    return RemoteTestConfig()
