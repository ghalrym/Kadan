from typing import Literal
from pydantic import BaseModel, model_validator

ModelType = Literal["LLM", "Image", "Video", "TTS", "STT"]


class ModelSetting(BaseModel):
    label: str
    type: ModelType
    selected: str | None
    options: list[str]

    @model_validator(mode="after")
    def selected_model_is_available(self) -> "ModelSetting":
        if self.selected is not None and self.selected not in self.options:
            raise ValueError("Selected model must be one of the available options")
        return self
