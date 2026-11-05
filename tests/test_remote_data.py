import pytest
from pydantic import ValidationError

from tests.support.remote_data import (
    ACCOUNT_URL_ENV,
    CONNECTION_STRING_ENV,
    CONTAINER_ENV,
    PREFIX_ENV,
    REMOTE_TESTS_ENV,
    RemoteTestConfig,
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
    assert config.connection_string.get_secret_value() == CONNECTION_STRING
    assert config.container_name == "test-container"
    assert config.prefix == "test-data"


@pytest.mark.parametrize("connection_string", [None, ""])
def test_load_remote_config_connection_string_is_optional(
    monkeypatch: pytest.MonkeyPatch, connection_string: str | None
):
    """Without a connection string the config falls back to DefaultAzureCredential."""
    _set_env(monkeypatch, connection_string=connection_string)

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

    assert config.connection_string.get_secret_value() == connection_string
    assert "fake-signature" not in repr(config)
    assert "SharedAccessSignature" not in repr(config)
    with BlobServiceClient.from_connection_string(config.connection_string.get_secret_value()) as client:
        assert client.account_name == "testaccount"


def test_remote_config_repr_hides_account_key(monkeypatch: pytest.MonkeyPatch):
    """Config representations must not expose account-key credentials either."""
    _set_env(monkeypatch)

    config = load_remote_config()

    assert "AccountKey" not in repr(config)
    assert "AccountKey" not in str(config.connection_string)
    assert "connection_string" not in config.model_dump()
    assert "AccountKey" not in config.model_dump_json()


def test_remote_config_explicit_values_override_environment(monkeypatch: pytest.MonkeyPatch):
    """Explicit settings win over environment values without changing them."""
    _set_env(monkeypatch)

    config = RemoteTestConfig(
        account_url="https://other.blob.core.windows.net",
        container_name="other-container",
        connection_string=None,
        prefix="/other-data/",
    )

    assert config.account_url == "https://other.blob.core.windows.net"
    assert config.container_name == "other-container"
    assert config.connection_string is None
    assert config.prefix == "other-data"
    assert load_remote_config().container_name == "test-container"


def test_remote_config_is_frozen(monkeypatch: pytest.MonkeyPatch):
    """The settings retain the previous dataclass's immutability."""
    _set_env(monkeypatch)
    config = load_remote_config()

    with pytest.raises(ValidationError, match="frozen"):
        config.prefix = "changed"


def test_remote_config_does_not_load_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Offline test configuration cannot pick up credentials from an implicit .env file."""
    _set_env(monkeypatch, connection_string=None)
    (tmp_path / ".env").write_text("AZURE_STORAGE_CONNECTION_STRING=private-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert load_remote_config().connection_string is None


def test_remote_config_validation_error_hides_rejected_secret(monkeypatch: pytest.MonkeyPatch):
    """Validation errors must not echo a secret rejected before SecretStr conversion."""
    _set_env(monkeypatch)

    with pytest.raises(ValidationError) as error:
        RemoteTestConfig(connection_string={"token": "private-token"})

    assert "private-token" not in str(error.value)


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


@pytest.mark.parametrize("variable, field_name", [(ACCOUNT_URL_ENV, "account_url"), (CONTAINER_ENV, "container_name")])
@pytest.mark.parametrize("value", [None, "", " "])
def test_load_remote_config_requires_account_and_container(
    monkeypatch: pytest.MonkeyPatch, variable: str, field_name: str, value: str | None
):
    """Missing required settings fail clearly without echoing other input values."""
    _set_env(monkeypatch, connection_string="private-token")
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match=field_name) as error:
        load_remote_config()

    assert "private-token" not in str(error.value)


def test_remote_config_uses_only_test_prefixed_location_settings(monkeypatch: pytest.MonkeyPatch):
    """Library settings and the previous container variable cannot configure cloud tests."""
    _set_env(monkeypatch, account_url=None, container=None, prefix=None, connection_string=None)
    monkeypatch.setenv("PYTHERMONDT_ACCOUNT_URL", ACCOUNT_URL)
    monkeypatch.setenv("PYTHERMONDT_CONTAINER_NAME", "library-container")
    monkeypatch.setenv("PYTHERMONDT_PREFIX", "library-prefix")
    monkeypatch.setenv("TEST_CONTAINER", "old-container")

    with pytest.raises(ValidationError) as error:
        load_remote_config()

    assert {item["loc"] for item in error.value.errors()} == {("account_url",), ("container_name",)}

    monkeypatch.setenv("TEST_ACCOUNT_URL", ACCOUNT_URL)
    monkeypatch.setenv("TEST_CONTAINER_NAME", "test-container")
    config = load_remote_config()

    assert config.container_name == "test-container"
    assert config.prefix == ""
