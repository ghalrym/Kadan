"""Decision route units replace queue boundaries; no service or inference."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1 import decisions


def body(**changes):
    return {
        "state": "A support ticket.",
        "questions": [
            {
                "key": "route",
                "type": "Choice",
                "instructions": "Route?",
                "options": [{"key": "billing", "description": "Payments"}],
            }
        ],
        **changes,
    }


class DecisionValidationTests(unittest.TestCase):
    def test_omitted_and_explicit_laya_preserve_the_native_contract(self):
        request = decisions.DecisionRequest(**body())
        self.assertIsNone(request.model)
        self.assertNotIn("model", request.model_dump(exclude_unset=True))
        self.assertEqual(decisions.DecisionRequest(**body(model="laya")).model, "laya")
        for changes in ({"state": "x" * 8001}, {"model": "chat-model"}):
            with self.assertRaises(ValidationError):
                decisions.DecisionRequest(**body(**changes))

    def test_d1_retains_full_state_and_rejects_unsupported_question_semantics(self):
        request = decisions.DecisionRequest(
            **body(model="d1-2b-bf16", state="x" * 20000)
        )
        self.assertEqual(len(request.state), 20000)
        queued = InferenceTask(
            input={"feature": "decisions", **request.model_dump(mode="json")}
        )
        self.assertEqual(len(queued.input.state), 20000)
        self.assertEqual(queued.input.model, "d1-2b-bf16")
        for levels in (["single"], [str(index) for index in range(11)]):
            questions = [
                {
                    "key": "score",
                    "type": "Score",
                    "instructions": "Rate?",
                    "levels": levels,
                }
            ]
            decisions.DecisionRequest(**body(questions=questions))
            with self.assertRaisesRegex(ValidationError, "2 to 10"):
                decisions.DecisionRequest(
                    **body(model="d1-2b-bf16", questions=questions)
                )
        request = body(model="d1-2b-bf16")
        request["questions"][0]["multiple"] = True
        with self.assertRaisesRegex(ValidationError, "unsupported fields"):
            decisions.DecisionRequest(**request)


class DecisionRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_model_and_answers_cross_the_queue_without_changing_laya(
        self,
    ):
        answer = {
            "key": "route",
            "type": "Choice",
            "value": "billing",
            "confidence": 1.0,
            "probabilities": {"billing": 1.0},
        }

        async def await_queue(request, queued):
            return await queued

        for model in (None, "laya", "d1-2b-bf16"):
            with self.subTest(model=model):
                request = decisions.DecisionRequest(
                    **body(**({"model": model} if model else {}))
                )
                result = (
                    {"model": model, "answers": [answer]}
                    if model == "d1-2b-bf16"
                    else [answer]
                )
                submit = AsyncMock(return_value=result)
                with (
                    patch.object(decisions.inference, "submit", submit),
                    patch.object(decisions, "infer", await_queue),
                ):
                    response = await decisions.evaluate_decisions(request, object())
                submit.assert_awaited_once_with(request, feature="decisions")
                self.assertEqual(response.answers[0].value, "billing")

    async def test_d1_rejects_wrong_model_or_partial_worker_output(self):
        request = decisions.DecisionRequest(**body(model="d1-2b-bf16"))

        async def await_queue(request, queued):
            return await queued

        for result in (
            {"model": "laya", "answers": []},
            {"model": "d1-2b-bf16", "answers": []},
            None,
        ):
            with (
                self.subTest(result=result),
                patch.object(
                    decisions.inference, "submit", AsyncMock(return_value=result)
                ),
                patch.object(decisions, "infer", await_queue),
            ):
                with self.assertRaises(HTTPException) as error:
                    await decisions.evaluate_decisions(request, object())
                self.assertEqual(error.exception.status_code, 502)
