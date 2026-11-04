"""Configure access to private Azure test data.

Set ``PYTHERMONDT_TEST_DATA_URI`` to the HTTPS container/prefix URL. Authentication
uses ``AZURE_STORAGE_CONNECTION_STRING`` when set, otherwise ``DefaultAzureCredential``
(e.g. ``az login``). In CI, store the connection string as a secret; prefer an expiring,
read/list-only container SAS.

Cloud tests run in GitHub Actions or with ``PYTHERMONDT_REMOTE_TESTS=1``.
Pull requests without the connection-string secret skip them.
"""

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
TEST_DATA_URI_ENV = "PYTHERMONDT_TEST_DATA_URI"
REMOTE_TESTS_ENV = "PYTHERMONDT_REMOTE_TESTS"

_TEST_DATA_URI_FORMAT = "https://<account>.blob.core.windows.net/<container>/<prefix>"


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
        RuntimeError: If the test data URI is missing or malformed.
    """
    uri = os.environ.get(TEST_DATA_URI_ENV, "")
    error = f"Invalid {TEST_DATA_URI_ENV}. Expected a credential-free URL: {_TEST_DATA_URI_FORMAT}."
    try:
        parsed = urlparse(uri)
    except ValueError:
        raise RuntimeError(error) from None
    container_name, _, prefix = parsed.path.lstrip("/").partition("/")
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or not container_name
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(error)

    return RemoteTestConfig(
        account_url=f"{parsed.scheme}://{parsed.netloc}",
        connection_string=os.environ.get(CONNECTION_STRING_ENV) or None,
        container_name=container_name,
        prefix=prefix.strip("/"),
    )
