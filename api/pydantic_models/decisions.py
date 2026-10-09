from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


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
    # Redis payloads use field names; HTTP requests use the existing aliases.
    model_config = ConfigDict(populate_by_name=True)
    key: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    type: Literal["Noul"]
    true_when: str = Field(default="", alias="trueWhen")
    false_when: str = Field(default="", alias="falseWhen")


DecisionQuestion = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class ChoiceAnswer(BaseModel):
    key: str
    type: Literal["Choice"]
    value: str
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    probabilities: dict[str, float] | None = None


class ScoreAnswer(BaseModel):
    key: str
    type: Literal["Score"]
    value: float = Field(ge=0, allow_inf_nan=False, description="Expected zero-based ordinal rubric index; may be fractional.")
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    probabilities: dict[str, float] | None = None


class NoulAnswer(BaseModel):
    key: str
    type: Literal["Noul"]
    value: float = Field(ge=0, le=1, allow_inf_nan=False, description="Probability that the criterion is true.")
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    probabilities: dict[str, float] | None = None


DecisionAnswer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]
