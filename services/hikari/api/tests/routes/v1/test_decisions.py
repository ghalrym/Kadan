import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.routes.v1.decisions import DecisionRequest, evaluate_decisions


class DecisionRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_d1_score_boundaries_are_explicit_and_laya_is_unchanged(self):
        for count in (1, 2, 10, 11):
            question = {
                "type": "Score",
                "key": "rating",
                "instructions": "Rate urgency",
                "levels": [f"level {i}" for i in range(count)],
            }
            with self.subTest(count=count, model="Laya"):
                self.assertEqual(
                    len(
                        DecisionRequest(state="text", questions=[question])
                        .questions[0]
                        .levels
                    ),
                    count,
                )
            if count in (2, 10):
                with self.subTest(count=count, model="d1"):
                    self.assertEqual(
                        len(
                            DecisionRequest(
                                state="text",
                                questions=[question],
                                model_id="d1-2b-cpu-f32",
                            )
                            .questions[0]
                            .levels
                        ),
                        count,
                    )
            else:
                with (
                    self.subTest(count=count, model="d1"),
                    self.assertRaises(ValidationError) as error,
                ):
                    DecisionRequest(
                        state="text", questions=[question], model_id="d1-2b-cpu-f32"
                    )
                self.assertEqual(
                    error.exception.errors()[0]["type"], "unsupported_d1_score_levels"
                )
                self.assertIn("d1 requires 2 to 10 Score levels", str(error.exception))

    async def test_explicit_trial_selection_is_forwarded_and_default_is_omitted(self):
        for fields in ({}, {"model_id": "d1-2b-cpu-f32"}):
            with self.subTest(fields=fields):
                body = DecisionRequest(
                    state="text",
                    questions=[{"type": "Noul", "key": "q", "instructions": "True?"}],
                    **fields,
                )

                async def finish(request, result):
                    return await result

                with (
                    patch(
                        "api.routes.v1.decisions.inference.submit",
                        new_callable=AsyncMock,
                        return_value=[],
                    ) as submit,
                    patch("api.routes.v1.decisions.infer", side_effect=finish),
                ):
                    result = await evaluate_decisions(body, object())
                self.assertEqual(result.answers, [])
                submitted = submit.call_args.args[0].model_dump(exclude_unset=True)
                self.assertEqual(submitted.get("model_id"), fields.get("model_id"))
                self.assertEqual("model_id" in submitted, bool(fields))

    def test_unrecognized_checkpoint_is_rejected(self):
        with self.assertRaises(ValidationError):
            DecisionRequest(
                state="text",
                questions=[{"type": "Noul", "key": "q", "instructions": "True?"}],
                model_id="another-model",
            )
