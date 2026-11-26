import pytest
from pydantic import ValidationError

from kadan.api.config import Settings


@pytest.mark.parametrize("field", ["tsuchi_url", "hikari_url"])
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
