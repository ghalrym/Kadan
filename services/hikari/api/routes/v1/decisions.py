from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from api.pydantic_models.decisions import (
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionQuestion,
    ScoreQuestion,
)
from api.services.inference import inference
from api.services.inference_http import infer

router = APIRouter(prefix="/v1/decisions", tags=["Decisions"])


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: DecisionModel | None = None
    state: str = Field(min_length=1)
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def d1_questions(cls, values):
        if isinstance(values, dict) and values.get("model") == "d1-2b-bf16":
            from experimental_worker.d1 import prepare_questions

            prepare_questions(values)
        return values

    @model_validator(mode="after")
    def validate_questions(self):
        """Validate request-wide keys, nonblank content and serialized size.

        Returns this parsed request; raises typed validation errors for ambiguous question/option
        keys, blank instructions or levels, or a request beyond the size budget."""
        keys = [question.key for question in self.questions]
        if any(not key.strip() for key in keys):
            raise PydanticCustomError(
                "blank_question_key", "Question keys must not be blank"
            )
        for index, key in enumerate(keys):
            if key in keys[:index]:
                raise PydanticCustomError(
                    "duplicate_question_key",
                    'Question key "{key}" is used more than once; each question must have a unique key',
                    {"key": key},
                )
        for question in self.questions:
            if not question.instructions.strip():
                raise PydanticCustomError(
                    "blank_instructions",
                    'Instructions for question "{key}" must not be blank',
                    {"key": question.key},
                )
            if isinstance(question, ChoiceQuestion):
                options = [option.key for option in question.options]
                if any(not option.strip() for option in options):
                    raise PydanticCustomError(
                        "blank_option_key",
                        'Option keys for question "{key}" must not be blank',
                        {"key": question.key},
                    )
                for index, option in enumerate(options):
                    if option in options[:index]:
                        raise PydanticCustomError(
                            "duplicate_option_key",
                            'Option key "{option}" is used more than once in question "{key}"',
                            {"option": option, "key": question.key},
                        )
            if isinstance(question, ScoreQuestion) and any(
                not level.strip() for level in question.levels
            ):
                raise PydanticCustomError(
                    "blank_score_level",
                    'Rubric levels for question "{key}" must not be blank',
                    {"key": question.key},
                )
        if not self.state.strip():
            raise PydanticCustomError("blank_state", "State must not be blank")
        if self.model != "d1-2b-bf16" and len(self.state) > 8000:
            raise PydanticCustomError(
                "state_too_large", "Laya state must fit 8000 characters"
            )
        if (
            self.model != "d1-2b-bf16"
            and len(self.model_dump_json(exclude={"model"})) > 16000
        ):
            raise PydanticCustomError(
                "request_too_large", "The total request must fit 16000 characters"
            )
        return self


class DecisionResponse(BaseModel):
    answers: list[DecisionAnswer]


class DecisionPlaygroundResponse(BaseModel):
    state: str
    questions: list[DecisionQuestion]
    answers: list[DecisionAnswer]


@router.get("", operation_id="getDecisions")
def get_decisions() -> DecisionPlaygroundResponse:
    """Return blank playground state; no saved questions or model answers are loaded."""
    return DecisionPlaygroundResponse(state="", questions=[], answers=[])


@router.post("", operation_id="evaluateDecisions")
async def evaluate_decisions(
    body: DecisionRequest, request: Request
) -> DecisionResponse:
    """Queue the selected specialist, preserving validation and disconnect cleanup."""
    try:
        result = await infer(request, inference.submit(body, feature="decisions"))
        if body.model == "d1-2b-bf16":
            if result["model"] != body.model:
                raise ValueError("The worker returned a different decision model.")
            result = result["answers"]
        response = DecisionResponse(answers=result)
        if body.model == "d1-2b-bf16" and [
            (answer.key, answer.type) for answer in response.answers
        ] != [(question.key, question.type) for question in body.questions]:
            raise ValueError("The worker returned different decision questions.")
        return response
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            502, "The decision model returned an invalid typed decision response."
        ) from exc
