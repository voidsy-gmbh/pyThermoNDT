"""Configuration for tests that read real cloud storage.

Integration test data lives in Azure Blob Storage instead of the repository. The tests
read two environment variables:

- ``AZURE_STORAGE_CONNECTION_STRING``: connection string of the test data storage account.
- ``PYTHERMONDT_TEST_DATA_URI``: test data location as ``az://<container>/<prefix>``.

The container mirrors the old asset layout under the prefix, for example
``test-data/integration/simulation/source1.mat`` next to
``test-data/integration/simulation/expected1.hdf5``.

Cloud-marked tests run in GitHub Actions and when ``PYTHERMONDT_REMOTE_TESTS=1`` is set.
Pull requests from forks skip cloud tests because they cannot read the connection string
secret.
"""

import os
from dataclasses import dataclass

CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
TEST_DATA_URI_ENV = "PYTHERMONDT_TEST_DATA_URI"
REMOTE_TESTS_ENV = "PYTHERMONDT_REMOTE_TESTS"

_TEST_DATA_URI_FORMAT = "az://<container>/<prefix>"


@dataclass(frozen=True)
class RemoteTestConfig:
    """Connection settings for the remote test data container.

    Attributes:
        account_url: Storage account URL (https://<account>.blob.core.windows.net).
        connection_string: Azure connection string used for authentication.
        container_name: Container that holds the test data.
        prefix: Prefix (folder path) of the test data within the container.
    """

    account_url: str
    connection_string: str
    container_name: str
    prefix: str


def remote_tests_enabled() -> bool:
    """Return True when cloud-marked tests should run.

    Cloud tests always run in GitHub Actions. Everywhere else they need an explicit
    opt-in through ``PYTHERMONDT_REMOTE_TESTS=1``.
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
        RuntimeError: If an environment variable is missing or malformed.
    """
    connection_string = os.environ.get(CONNECTION_STRING_ENV, "")
    if not connection_string:
        raise RuntimeError(
            f"{CONNECTION_STRING_ENV} is not set. Provide a connection string for the test data account."
        )

    uri = os.environ.get(TEST_DATA_URI_ENV, "")
    if not uri.startswith("az://") or not uri[len("az://") :].partition("/")[0]:
        raise RuntimeError(f"Invalid {TEST_DATA_URI_ENV}={uri!r}. Expected format: {_TEST_DATA_URI_FORMAT}.")
    container_name, _, prefix = uri[len("az://") :].partition("/")

    return RemoteTestConfig(
        account_url=_account_url_from_connection_string(connection_string),
        connection_string=connection_string,
        container_name=container_name,
        prefix=prefix.strip("/"),
    )


def _account_url_from_connection_string(connection_string: str) -> str:
    """Extract the blob account URL from a connection string.

    Raises:
        RuntimeError: If the connection string has no AccountName entry.
    """
    for part in connection_string.split(";"):
        key, _, value = part.partition("=")
        if key == "AccountName" and value:
            return f"https://{value}.blob.core.windows.net"
    raise RuntimeError(f"Cannot derive the account URL: {CONNECTION_STRING_ENV} has no AccountName entry.")
