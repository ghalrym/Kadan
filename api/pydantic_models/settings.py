from typing import Literal
from pydantic import BaseModel, model_validator

ModelType = Literal["LLM", "Image", "Video", "TTS", "STT"]


class ModelSetting(BaseModel):
    """A catalog-backed selection; None means no model is selected."""
    label: str
    type: ModelType
    selected: str | None
    options: list[str]

    @model_validator(mode="after")
    def selected_model_is_available(self) -> "ModelSetting":
        """Return this setting after checking selection membership.

        Raise ValueError for a non-null selection absent from the advertised options."""
        if self.selected is not None and self.selected not in self.options:
            raise ValueError("Selected model must be one of the available options")
        return self
