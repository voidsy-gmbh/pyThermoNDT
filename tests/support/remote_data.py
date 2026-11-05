"""Configure access to private Azure test data.

Set ``TEST_ACCOUNT_URL`` and ``TEST_CONTAINER``; optionally set ``TEST_PREFIX``
(default: container root). These are passed to AzureBlobReader
without URL splitting. Keep credentials in ``AZURE_STORAGE_CONNECTION_STRING``, not the
account URL. Without a connection string, authentication uses ``DefaultAzureCredential``
(e.g. ``az login``). In CI, prefer an expiring, read/list-only container SAS secret.

Cloud tests run in GitHub Actions or with ``REMOTE_TESTS=1``.
Pull requests without the connection-string secret skip them.
"""

import os
from dataclasses import dataclass, field

CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
ACCOUNT_URL_ENV = "TEST_ACCOUNT_URL"
CONTAINER_ENV = "TEST_CONTAINER"
PREFIX_ENV = "TEST_PREFIX"
REMOTE_TESTS_ENV = "REMOTE_TESTS"


@dataclass(frozen=True)
class RemoteTestConfig:
    """Connection settings for the remote test data container.

    Attributes:
        account_url: Storage account URL (https://<account>.blob.core.windows.net).
        connection_string: Azure connection string used for authentication, or None to
            fall back to DefaultAzureCredential (e.g. ``az login``).
        container_name: Container that holds the test data.
        prefix: Prefix (folder path) of the test data within the container.
    """

    account_url: str
    connection_string: str | None = field(repr=False)
    container_name: str
    prefix: str


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
        RuntimeError: If the account URL or container name is missing.
    """
    account_url = os.environ.get(ACCOUNT_URL_ENV, "")
    container_name = os.environ.get(CONTAINER_ENV, "")
    if not account_url.strip():
        raise RuntimeError(f"{ACCOUNT_URL_ENV} is required for remote test data.")
    if not container_name.strip():
        raise RuntimeError(f"{CONTAINER_ENV} is required for remote test data.")

    return RemoteTestConfig(
        account_url=account_url,
        connection_string=os.environ.get(CONNECTION_STRING_ENV) or None,
        container_name=container_name,
        prefix=os.environ.get(PREFIX_ENV, "").strip("/"),
    )
