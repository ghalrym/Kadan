from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KADAN_", env_file=".env", extra="ignore"
    )

    allowed_hosts: list[str] = ["127.0.0.1", "localhost", "::1"]
    tsuchi_url: str = "http://127.0.0.1:8620"
    hikari_url: str = "http://127.0.0.1:8088"
    tsuchi_browser_url: str = "http://127.0.0.1:8620"
    hikari_browser_url: str = "http://127.0.0.1:8088"
    # The agent's AIO sandbox computer. Unset leaves the agent without one.
    sandbox_url: str | None = None
    sandbox_api_key: SecretStr | None = None
    frontend_directory: str = "web/dist"
    agent_instruction_file: str = str(Path(__file__).resolve().parents[2] / "AGENT.md")

    @field_validator(
        "tsuchi_url",
        "hikari_url",
        "tsuchi_browser_url",
        "hikari_browser_url",
        "sandbox_url",
    )
    @classmethod
    def local_service_url(cls, value: str | None, info: ValidationInfo) -> str | None:
        if info.field_name == "sandbox_url" and not value:
            return None
        internal_origins = {
            "tsuchi_url": "http://tsuchi-api:8000",
            "hikari_url": "http://hikari-api:8000",
            "sandbox_url": "http://kadan-sandbox:8080",
        }
        origin = value.removesuffix("/")
        if origin == internal_origins.get(info.field_name):
            return origin
        allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        if info.field_name in {"tsuchi_browser_url", "hikari_browser_url"}:
            allowed_hosts.update(
                host.lower().strip("[]") for host in info.data.get("allowed_hosts", [])
            )
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or any(character.isspace() for character in value)
        ):
            raise ValueError(
                "Use a loopback HTTP origin, this service's exact Docker origin, "
                "or an explicitly allowed host for browser URLs"
            )
        if parsed.port == 0:
            raise ValueError("A service port must be between 1 and 65535")
        return origin


class ServiceError(Exception):
    def __init__(
        self, message: str, status: int = 503, code: str = "service_unavailable"
    ):
        super().__init__(message)
        self.status = status
        self.code = code
