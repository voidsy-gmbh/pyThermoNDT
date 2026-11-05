import pytest

from tests.support.remote_data import (
    ACCOUNT_URL_ENV,
    CONNECTION_STRING_ENV,
    CONTAINER_ENV,
    PREFIX_ENV,
    REMOTE_TESTS_ENV,
    cloud_skip_reason,
    load_remote_config,
    remote_tests_enabled,
)

CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=abc==;EndpointSuffix=core.windows.net"
)
ACCOUNT_URL = "https://testaccount.blob.core.windows.net"


def _set_env(
    monkeypatch: pytest.MonkeyPatch,
    connection_string: str | None = CONNECTION_STRING,
    account_url: str | None = ACCOUNT_URL,
    container: str | None = "test-container",
    prefix: str | None = "test-data",
):
    """Set the remote test environment variables; None deletes a variable."""
    for name, value in (
        (CONNECTION_STRING_ENV, connection_string),
        (ACCOUNT_URL_ENV, account_url),
        (CONTAINER_ENV, container),
        (PREFIX_ENV, prefix),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "github_actions, remote_tests, expected",
    [
        (None, None, False),
        (None, "1", True),
        (None, "true", True),
        (None, "on", True),
        (None, "0", False),
        (None, "false", False),
        ("true", None, True),
    ],
)
def test_remote_tests_enabled(
    monkeypatch: pytest.MonkeyPatch, github_actions: str | None, remote_tests: str | None, expected: bool
):
    """Cloud tests run in GitHub Actions or after an explicit opt-in."""
    if github_actions is None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    else:
        monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
    if remote_tests is None:
        monkeypatch.delenv(REMOTE_TESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(REMOTE_TESTS_ENV, remote_tests)

    assert remote_tests_enabled() is expected


def test_skip_reason_when_disabled(monkeypatch: pytest.MonkeyPatch):
    """Without opt-in, cloud tests skip with the opt-in instructions."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv(REMOTE_TESTS_ENV, raising=False)

    reason = cloud_skip_reason()

    assert reason is not None and REMOTE_TESTS_ENV in reason


def test_skip_reason_for_fork_pull_request(monkeypatch: pytest.MonkeyPatch):
    """Pull requests without credentials skip instead of fail."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.delenv(CONNECTION_STRING_ENV, raising=False)

    reason = cloud_skip_reason()

    assert reason is not None and CONNECTION_STRING_ENV in reason


@pytest.mark.parametrize("github_event", ["pull_request", "push"])
def test_no_skip_reason_with_enabled_and_credentials(monkeypatch: pytest.MonkeyPatch, github_event: str):
    """Enabled cloud tests with credentials do not skip."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(REMOTE_TESTS_ENV, "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", github_event)
    _set_env(monkeypatch)

    assert cloud_skip_reason() is None


def test_load_remote_config_reads_separate_settings(monkeypatch: pytest.MonkeyPatch):
    """Account URL, container, prefix, and credentials are loaded independently."""
    _set_env(monkeypatch)

    config = load_remote_config()

    assert config.account_url == "https://testaccount.blob.core.windows.net"
    assert config.connection_string == CONNECTION_STRING
    assert config.container_name == "test-container"
    assert config.prefix == "test-data"


def test_load_remote_config_connection_string_is_optional(monkeypatch: pytest.MonkeyPatch):
    """Without a connection string the config falls back to DefaultAzureCredential."""
    _set_env(monkeypatch, connection_string=None)

    config = load_remote_config()

    assert config.connection_string is None
    assert config.container_name == "test-container"


def test_load_remote_config_accepts_sas_connection_string(monkeypatch: pytest.MonkeyPatch):
    """SAS connection strings reach the Azure SDK unchanged and stay out of repr."""
    from azure.storage.blob import BlobServiceClient

    connection_string = (
        "BlobEndpoint=https://testaccount.blob.core.windows.net;"
        "SharedAccessSignature=sv=2024-11-04&sr=c&sp=rl&sig=fake-signature"
    )
    _set_env(monkeypatch, connection_string=connection_string)

    config = load_remote_config()

    assert config.connection_string == connection_string
    assert "fake-signature" not in repr(config)
    assert "SharedAccessSignature" not in repr(config)
    with BlobServiceClient.from_connection_string(config.connection_string) as client:
        assert client.account_name == "testaccount"


def test_remote_config_repr_hides_account_key(monkeypatch: pytest.MonkeyPatch):
    """Config representations must not expose account-key credentials either."""
    _set_env(monkeypatch)

    assert "AccountKey" not in repr(load_remote_config())


@pytest.mark.parametrize("prefix", [None, "", "/"])
def test_load_remote_config_prefix_is_optional(monkeypatch: pytest.MonkeyPatch, prefix: str | None):
    """An unset or empty prefix selects the container root."""
    _set_env(monkeypatch, prefix=prefix)

    config = load_remote_config()

    assert config.container_name == "test-container"
    assert config.prefix == ""


def test_load_remote_config_preserves_account_and_blob_prefix(monkeypatch: pytest.MonkeyPatch):
    """The account endpoint is not reconstructed and blob prefix characters stay intact."""
    _set_env(monkeypatch, account_url=ACCOUNT_URL + ":8443/", prefix="/test data/measurements%20one/")

    config = load_remote_config()

    assert config.account_url == ACCOUNT_URL + ":8443/"
    assert config.prefix == "test data/measurements%20one"


@pytest.mark.parametrize("variable", [ACCOUNT_URL_ENV, CONTAINER_ENV])
@pytest.mark.parametrize("value", [None, "", " "])
def test_load_remote_config_requires_account_and_container(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str | None
):
    """Missing required settings fail clearly without echoing other input values."""
    _set_env(monkeypatch, connection_string="private-token")
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)

    with pytest.raises(RuntimeError, match=variable) as error:
        load_remote_config()

    assert "private-token" not in str(error.value)
