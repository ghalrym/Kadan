from collections.abc import Iterator
from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL, Engine, create_engine, event
from sqlalchemy.orm import Session


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", env_file=".env", extra="ignore"
    )

    host: str = "postgres"
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = "kadan"
    user: str = "kadan"
    password: SecretStr = SecretStr("kadan")

    @property
    def url(self) -> URL | str:
        explicit = os.environ.get("HIKARI_DATABASE_URL", "").strip()
        if explicit:
            return explicit
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.db,
        )


def _is_sqlite(url: URL | str) -> bool:
    return str(url).startswith("sqlite")


def _ensure_sqlite_parent(url: URL | str) -> None:
    raw = str(url)
    for prefix in ("sqlite+pysqlite:///", "sqlite:///"):
        if raw.startswith(prefix):
            path_part = raw[len(prefix) :].split("?", 1)[0]
            if path_part and path_part != ":memory:":
                Path(path_part).parent.mkdir(parents=True, exist_ok=True)
            return


def build_engine(url: URL | str) -> Engine:
    if _is_sqlite(url):
        _ensure_sqlite_parent(url)
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    return build_engine(DatabaseSettings().url)


def ensure_database_ready() -> None:
    """Create or migrate tables when using SQLite or when HIKARI_AUTO_MIGRATE is set."""
    url = DatabaseSettings().url
    auto = os.environ.get("HIKARI_AUTO_MIGRATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not auto and not _is_sqlite(url):
        return
    from alembic import command
    from alembic.config import Config

    alembic_ini = Path(__file__).resolve().parent / "alembic.ini"
    cfg = Config(str(alembic_ini))
    cfg.set_main_option(
        "script_location", str(Path(__file__).resolve().parent / "migrations")
    )
    command.upgrade(cfg, "head")


def get_session() -> Iterator[Session]:
    """Provide a session; callers explicitly commit successful writes."""
    with Session(get_engine()) as session:
        yield session
