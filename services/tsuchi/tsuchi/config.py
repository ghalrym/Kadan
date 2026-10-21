from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TSUCHI_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Path("data")
    static_dir: Path = Path("web/dist")
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, gt=0)


class StorageError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)
