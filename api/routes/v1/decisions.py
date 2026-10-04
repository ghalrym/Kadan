from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, model_validator
from api.pydantic_models.decisions import (
    ChoiceOption, ChoiceQuestion, DecisionAnswer, DecisionQuestion, NoulQuestion, ScoreQuestion,
)

router = APIRouter(prefix="/v1/decisions", tags=["Decisions"])

MOCK_STATE = ('Image generation has returned HTTP 500 for the last hour. Logs show "CUDA out of '
 'memory while allocating 2.1 GiB" on GPU 0. Batch jobs from batch-worker are stuck '
 'and customers are waiting on renders.')

MOCK_QUESTIONS: list[DecisionQuestion] = [
    ChoiceQuestion(
        key='cause',
        type='Choice',
        instructions='What is most likely causing the failures',
        options=[
            ChoiceOption(
                key='gpu_capacity',
                description='Out of memory or GPU saturation',
            ),
            ChoiceOption(
                key='model_bug',
                description='Model returns wrong or malformed output',
            ),
            ChoiceOption(
                key='client_error',
                description='Bad requests from the caller',
            ),
            ChoiceOption(
                key='network',
                description='Timeouts or connectivity problems',
            ),
        ],
    ),
    ScoreQuestion(
        key='severity',
        type='Score',
        instructions='How severe the impact on users is',
        levels=['Minor inconvenience', 'Degraded service', 'Full outage'],
    ),
    NoulQuestion(
        key='is_urgent',
        type='Noul',
        instructions='The message conveys urgency or time-sensitivity',
        trueWhen='',
        falseWhen='',
    ),
]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str = Field(min_length=1)
    questions: list[DecisionQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_question_keys(self) -> "DecisionRequest":
        keys = [question.key for question in self.questions]
        if len(keys) != len(set(keys)):
            raise ValueError("Question keys must be unique")
        return self


class DecisionResponse(BaseModel):
    answers: list[DecisionAnswer]


class DecisionPlaygroundResponse(BaseModel):
    state: str
    questions: list[DecisionQuestion]
    answers: list[DecisionAnswer]


@router.get("", operation_id="getDecisions")
def get_decisions() -> DecisionPlaygroundResponse:
    return DecisionPlaygroundResponse(state=MOCK_STATE, questions=MOCK_QUESTIONS, answers=[])


@router.post("", operation_id="evaluateDecisions")
def evaluate_decisions(body: DecisionRequest) -> DecisionResponse:
    # The mock playground has no evaluated answers yet.
    return DecisionResponse(answers=[])
