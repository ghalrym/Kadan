import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.routes.v1.decisions import DecisionRequest, evaluate_decisions


class DecisionRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_score_rubrics_require_two_to_ten_levels(self):
        for count in (0, 1, 2, 10, 11, 64):
            question = {
                "type": "Score",
                "key": "rating",
                "instructions": "Rate urgency",
                "levels": [f"level {i}" for i in range(count)],
            }
            with self.subTest(count=count):
                if count in (2, 10):
                    self.assertEqual(
                        len(
                            DecisionRequest(state="text", questions=[question])
                            .questions[0]
                            .levels
                        ),
                        count,
                    )
                else:
                    with self.assertRaises(ValidationError) as error:
                        DecisionRequest(state="text", questions=[question])
                    self.assertEqual(
                        error.exception.errors()[0]["loc"],
                        ("questions", 0, "Score", "levels"),
                    )
                    self.assertIn(
                        "2" if count < 2 else "10", error.exception.errors()[0]["msg"]
                    )

    async def test_default_evaluation_submits_without_a_public_model_selector(self):
        body = DecisionRequest(
            state="text",
            questions=[{"type": "Noul", "key": "q", "instructions": "True?"}],
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
        self.assertEqual(submit.call_args.kwargs, {"feature": "decisions"})
        self.assertNotIn("model_id", submit.call_args.args[0].model_dump())

    def test_model_selection_is_rejected_including_former_trial_selector(self):
        for model in ("laya", "d1-2b-cpu-f32", "another-model", None):
            with self.subTest(model=model), self.assertRaises(ValidationError) as error:
                DecisionRequest(
                    state="text",
                    questions=[{"type": "Noul", "key": "q", "instructions": "True?"}],
                    model_id=model,
                )
            self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")
