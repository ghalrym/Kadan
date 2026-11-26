from pathlib import Path
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KADAN_", env_file=".env", extra="ignore"
    )

    tsuchi_url: str = "http://127.0.0.1:8620"
    hikari_url: str = "http://127.0.0.1:8088"
    tsuchi_browser_url: str = "http://127.0.0.1:8620"
    hikari_browser_url: str = "http://127.0.0.1:8088"
    frontend_directory: str = "web/dist"
    agent_instruction_file: str = str(Path(__file__).resolve().parents[2] / "AGENT.md")

    @field_validator(
        "tsuchi_url",
        "hikari_url",
        "tsuchi_browser_url",
        "hikari_browser_url",
    )
    @classmethod
    def service_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Use an HTTP or HTTPS service URL")
        if parsed.port == 0:
            raise ValueError("A service port must be between 1 and 65535")
        return value.rstrip("/")


class ServiceError(Exception):
    def __init__(
        self, message: str, status: int = 503, code: str = "service_unavailable"
    ):
        super().__init__(message)
        self.status = status
        self.code = code
