from typing import Annotated, Literal
from pydantic import BaseModel, Field


class ChoiceOption(BaseModel):
    key: str = Field(min_length=1)
    description: str


class ChoiceQuestion(BaseModel):
    key: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    type: Literal["Choice"]
    options: list[ChoiceOption] = Field(min_length=1)


class ScoreQuestion(BaseModel):
    key: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    type: Literal["Score"]
    levels: list[str] = Field(min_length=1)


class NoulQuestion(BaseModel):
    key: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    type: Literal["Noul"]
    true_when: str = Field(default="", alias="trueWhen")
    false_when: str = Field(default="", alias="falseWhen")


DecisionQuestion = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class DecisionAnswer(BaseModel):
    key: str
    type: Literal["Choice", "Score", "Noul"]
    value: str | int | float | bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, float] | None = None
