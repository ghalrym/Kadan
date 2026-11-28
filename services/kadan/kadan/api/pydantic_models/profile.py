"""The person using Kadan: their name and pet."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .pets import PetRef


class UserProfile(BaseModel):
    display_name: str = "You"
    pet: PetRef | None = None
    version: int = Field(default=0, ge=0)
    updated_at: datetime | None = None


class UserProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    pet: PetRef | None = None
    clear_pet: bool = False
    expected_version: int = Field(ge=0)

    @field_validator("display_name")
    @classmethod
    def nonblank_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def one_pet_change(self) -> Self:
        if self.pet is not None and self.clear_pet:
            raise ValueError("Choose a pet or clear it, not both")
        return self
