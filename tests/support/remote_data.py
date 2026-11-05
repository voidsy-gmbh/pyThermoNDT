"""Configure access to private Azure test data.

Set ``TEST_ACCOUNT_URL`` and ``TEST_CONTAINER_NAME``; optionally set ``TEST_PREFIX``
(default: container root). These are passed to AzureBlobReader
without URL splitting. Keep credentials in ``AZURE_STORAGE_CONNECTION_STRING``, not the
account URL. Without a connection string, authentication uses ``DefaultAzureCredential``
(e.g. ``az login``). In CI, prefer an expiring, read/list-only container SAS secret.

Cloud tests run in GitHub Actions or with ``REMOTE_TESTS=1``.
Pull requests without the connection-string secret skip them.
"""

import os

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
ACCOUNT_URL_ENV = "TEST_ACCOUNT_URL"
CONTAINER_ENV = "TEST_CONTAINER_NAME"
PREFIX_ENV = "TEST_PREFIX"
REMOTE_TESTS_ENV = "REMOTE_TESTS"


class RemoteTestConfig(BaseSettings):
    """Connection settings for the remote test data container.

    Attributes:
        account_url: Storage account URL (https://<account>.blob.core.windows.net).
        connection_string: Azure connection string used for authentication, or None to
            fall back to DefaultAzureCredential (e.g. ``az login``).
        container_name: Container that holds the test data.
        prefix: Prefix (folder path) of the test data within the container.
    """

    model_config = SettingsConfigDict(
        env_prefix="TEST_",
        frozen=True,
        populate_by_name=True,
        case_sensitive=False,
        env_ignore_empty=True,
        hide_input_in_errors=True,
    )

    account_url: str = Field(min_length=1, pattern=r"\S")
    connection_string: SecretStr | None = Field(
        default=None, validation_alias=CONNECTION_STRING_ENV, repr=False, exclude=True
    )
    container_name: str = Field(min_length=1, pattern=r"\S")
    prefix: str = ""

    @field_validator("prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        """Keep prefixes relative to the container, without changing blob-name characters."""
        return value.strip("/")


def remote_tests_enabled() -> bool:
    """Return True when cloud-marked tests should run.

    Cloud tests always run in GitHub Actions. Everywhere else they need an explicit
    opt-in through ``REMOTE_TESTS=1``.
    """
    opt_in = os.environ.get(REMOTE_TESTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    return opt_in or os.environ.get("GITHUB_ACTIONS") == "true"


def cloud_skip_reason() -> str | None:
    """Return why cloud-marked tests must skip, or None when they can run.

    Pull requests without a connection string skip instead of fail: forks cannot read
    repository secrets, and a missing secret must not break their checks.
    """
    if not remote_tests_enabled():
        return f"cloud tests disabled; set {REMOTE_TESTS_ENV}=1 to run them"
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and not os.environ.get(CONNECTION_STRING_ENV):
        return f"pull request without {CONNECTION_STRING_ENV}; fork pull requests cannot read secrets"
    return None


def load_remote_config() -> RemoteTestConfig:
    """Load the test data location and Azure credentials from the environment.

    Returns:
        Connection settings for the test data container.

    Raises:
        pydantic.ValidationError: If required settings are missing or invalid.
    """
    return RemoteTestConfig()
