import pytest
from pydantic import ValidationError

from kadan.api.config import Settings


@pytest.mark.parametrize(
    "field",
    [
        "tsuchi_url",
        "hikari_url",
        "tsuchi_browser_url",
        "hikari_browser_url",
        "sandbox_url",
    ],
)
@pytest.mark.parametrize(
    "url",
    [
        "http://tsuchi-api:8000",
        "http://192.0.2.10:8620/",
        "https://services.example.com/kadan/",
    ],
)
def test_service_urls_accept_remote_hosts_and_https(field, url):
    settings = Settings(**{field: url}, _env_file=None)
    assert getattr(settings, field) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "http://",
        "http://example.com:0",
        "http://example.com:65536",
    ],
)
def test_service_urls_reject_invalid_http_endpoints(url):
    with pytest.raises(ValidationError):
        Settings(tsuchi_url=url, _env_file=None)


@pytest.mark.parametrize("url", ["", None])
def test_computer_is_optional(url):
    assert Settings(sandbox_url=url, _env_file=None).sandbox_url is None


def test_sandbox_key_defaults_to_public_api_key(monkeypatch):
    monkeypatch.delenv("KADAN_SANDBOX_API_KEY", raising=False)
    assert (
        Settings(_env_file=None).sandbox_api_key.get_secret_value()
        == "insecure-api-key"
    )


def test_sandbox_key_can_be_overridden():
    assert (
        Settings(
            sandbox_api_key="custom-key", _env_file=None
        ).sandbox_api_key.get_secret_value()
        == "custom-key"
    )
