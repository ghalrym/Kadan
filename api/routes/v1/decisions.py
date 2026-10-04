import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from api.pydantic_models.decisions import ChoiceQuestion, DecisionAnswer, DecisionQuestion, ScoreQuestion
from api.services.runtime import RuntimeFailure
from api.services.decisions import decision_manager

router = APIRouter(prefix='/v1/decisions', tags=['Decisions'])


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    state: str = Field(min_length=1, max_length=8000)
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=8)

    @model_validator(mode='after')
    def validate_questions(self):
        """Validate request-wide keys, nonblank content and serialized size.

        Returns this parsed request; raises ValueError for ambiguous question/option
        keys, blank instructions or levels, or a request beyond the size budget."""
        keys = [question.key for question in self.questions]
        if len(keys) != len(set(keys)) or any(not key.strip() for key in keys):
            raise ValueError('Question keys must be nonblank and unique')
        for question in self.questions:
            if not question.instructions.strip():
                raise ValueError('Question instructions must not be blank')
            if isinstance(question, ChoiceQuestion):
                options = [option.key for option in question.options]
                if len(options) != len(set(options)) or any(not option.strip() for option in options):
                    raise ValueError('Choice option keys must be nonblank and unique')
            if isinstance(question, ScoreQuestion) and any(not level.strip() for level in question.levels):
                raise ValueError('Score levels must not be blank')
        if not self.state.strip() or len(self.model_dump_json()) > 16000:
            raise ValueError('State must be nonblank and total request must fit 16000 characters')
        return self


class DecisionResponse(BaseModel):
    answers: list[DecisionAnswer]


class DecisionPlaygroundResponse(BaseModel):
    state: str
    questions: list[DecisionQuestion]
    answers: list[DecisionAnswer]


@router.get('', operation_id='getDecisions')
def get_decisions() -> DecisionPlaygroundResponse:
    """Return blank playground state; no saved questions or model answers are loaded."""
    return DecisionPlaygroundResponse(state='', questions=[], answers=[])


@router.post('', operation_id='evaluateDecisions')
async def evaluate_decisions(body: DecisionRequest, request: Request) -> DecisionResponse:
    """Evaluate typed questions with the resident CPU Laya specialist, independent of chat.

    Oversized tokenized questions/state return 422 rather than truncated answers.
    Disconnect cancellation waits for the CPU worker before releasing ownership.
    Results are not persisted; there is no fallback to a generative model.
    """
    async def disconnect():
        """Wait for the ASGI client-disconnect event so inference can be cancelled."""
        while True:
            if (await request.receive())['type'] == 'http.disconnect':
                return
    generation = asyncio.create_task(decision_manager.evaluate(body.state, body.questions))
    disconnected = asyncio.create_task(disconnect())
    try:
        done, _ = await asyncio.wait([generation, disconnected], return_when=asyncio.FIRST_COMPLETED)
        if generation not in done:
            generation.cancel()
            raise HTTPException(499, 'Client disconnected; evaluation cancelled.')
        try:
            return DecisionResponse(answers=await generation)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(502, 'Laya returned an invalid typed decision response.') from exc
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    finally:
        for task in (generation, disconnected):
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
