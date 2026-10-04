from collections.abc import Iterator
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import Session


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "postgres"
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = "kadan"
    user: str = "kadan"
    password: SecretStr = SecretStr("kadan")

    @property
    def url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.db,
        )


@lru_cache
def get_engine() -> Engine:
    return create_engine(DatabaseSettings().url, pool_pre_ping=True)


def get_session() -> Iterator[Session]:
    """Provide a session; callers explicitly commit successful writes."""
    with Session(get_engine()) as session:
        yield session
