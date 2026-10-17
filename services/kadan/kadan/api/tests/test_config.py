import pytest
from pydantic import ValidationError

from kadan.api.config import Settings


@pytest.mark.parametrize(
    ("field", "url"),
    [
        ("tsuchi_url", "http://tsuchi-api:8000"),
        ("hikari_url", "http://hikari-api:8000"),
        ("tsuchi_url", "http://127.0.0.1:18620/"),
        ("hikari_url", "http://localhost:18088"),
        ("tsuchi_browser_url", "http://[::1]:8620"),
        ("hikari_browser_url", "http://127.0.0.1:8088/"),
    ],
)
def test_config_accepts_only_local_or_owned_service_origins(field, url):
    settings = Settings(**{field: url}, _env_file=None)
    assert getattr(settings, field) == url.removesuffix("/")


@pytest.mark.parametrize(
    ("field", "url"),
    [
        ("tsuchi_url", "http://hikari-api:8000"),
        ("hikari_url", "http://tsuchi-api:8000"),
        ("tsuchi_url", "http://tsuchi-api:8620"),
        ("hikari_url", "http://hikari-api"),
        ("tsuchi_url", "https://127.0.0.1:8620"),
        ("tsuchi_url", "http://example.com"),
        ("tsuchi_url", "http://127.0.0.1/path"),
        ("tsuchi_url", "http://user:secret@127.0.0.1"),
        ("tsuchi_url", "http://@127.0.0.1"),
        ("tsuchi_url", "http://127.0.0.1?url=x"),
        ("tsuchi_url", "http://127.0.0.1#fragment"),
        ("tsuchi_url", "http://127.0.0.1:65536"),
        ("tsuchi_url", "http://127.0.0.1:0"),
        ("tsuchi_url", "\nhttp://127.0.0.1"),
        ("tsuchi_url", "http://tsuchi-api:8000/path"),
        ("hikari_url", "http://hikari-api:8000?query=x"),
        ("tsuchi_browser_url", "http://tsuchi-api:8000"),
        ("hikari_browser_url", "http://hikari-api:8000"),
        ("hikari_browser_url", "http://192.168.1.10:8088"),
    ],
)
def test_config_rejects_unowned_origins_paths_and_credentials(field, url):
    with pytest.raises(ValidationError):
        Settings(**{field: url}, _env_file=None)


def test_explicit_lan_host_is_allowed_for_browser_links():
    settings = Settings(
        allowed_hosts=["localhost", "127.0.0.1", "192.0.2.10"],
        tsuchi_browser_url="http://192.0.2.10:8620/",
        hikari_browser_url="http://192.0.2.10:8088",
        _env_file=None,
    )
    assert settings.tsuchi_browser_url == "http://192.0.2.10:8620"
    assert settings.hikari_browser_url == "http://192.0.2.10:8088"


@pytest.mark.parametrize("field", ["tsuchi_url", "hikari_url"])
def test_browser_host_allowlist_does_not_expand_upstream_origins(field):
    with pytest.raises(ValidationError):
        Settings(
            allowed_hosts=["192.0.2.10"],
            **{field: "http://192.0.2.10:8000"},
            _env_file=None,
        )
