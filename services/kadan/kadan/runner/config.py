from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..api.pydantic_models.resources import AgentKey


class RunnerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KADAN_RUNNER_", env_file=".env", extra="ignore"
    )

    id: AgentKey = "runner-1"
    capacity: int = Field(default=5, ge=1, strict=True)

    @field_validator("capacity", mode="before")
    @classmethod
    def integer_capacity(cls, value):
        # Environment variables are strings; Python/JSON inputs stay strict.
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
        return value
