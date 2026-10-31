import importlib.util
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from experimental_worker.decision import evaluate


def task():
    return {
        "state": "The parcel is late.",
        "questions": [
            {
                "key": "route",
                "type": "Choice",
                "instructions": "Pick a route",
                "options": [
                    {"key": "billing", "description": "Charges"},
                    {"key": "shipping", "description": ""},
                ],
            },
            {
                "key": "priority",
                "type": "Score",
                "instructions": "Assess priority",
                "levels": ["Can wait", "Today", "Now"],
            },
            {
                "key": "urgent",
                "type": "Noul",
                "instructions": "Is it urgent?",
                "trueWhen": "Needs action now",
                "falseWhen": "Can wait",
            },
        ],
    }


def result():
    return {
        "answers": {
            "urgent": {"type": "noul", "noul": 0.8},
            "priority": {
                "type": "score",
                "score": 1.1,
                "confidence": 0.5,
                "probabilities": {"2": 0.3, "0": 0.2, "1": 0.5},
            },
            "route": {
                "type": "choice",
                "choice": "shipping",
                "confidence": 0.75,
                "probabilities": {"shipping": 0.75, "billing": 0.25},
            },
        },
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


class FakeModel:
    def __init__(self, output=None):
        self.output = result() if output is None else output
        self.calls = []

    def system_one(self, state, questions, *, score_readout):
        self.calls.append((state, questions, score_readout))
        return self.output


class DecisionTests(unittest.TestCase):
    def test_maps_questions_and_preserves_fractional_mean_and_request_order(self):
        model = FakeModel()
        request = task()
        original = deepcopy(request)
        answers = evaluate(request, model, lambda: None)
        self.assertEqual(request, original)
        self.assertEqual(
            [answer["key"] for answer in answers], ["route", "priority", "urgent"]
        )
        self.assertEqual(answers[1]["value"], 1.1)
        self.assertEqual(answers[0]["confidence"], 0.75)
        self.assertEqual(answers[2], {"key": "urgent", "type": "Noul", "value": 0.8})
        state, questions, readout = model.calls[0]
        self.assertEqual(state, request["state"])
        self.assertEqual(readout, "mean")
        self.assertEqual(list(questions), ["route", "priority", "urgent"])
        self.assertEqual(
            questions["route"]["criteria"], {"billing": "Charges", "shipping": ""}
        )
        self.assertEqual(
            questions["priority"]["criteria"], ["Can wait", "Today", "Now"]
        )
        self.assertEqual(
            questions["urgent"]["criteria"],
            {"true": "Needs action now", "false": "Can wait"},
        )

    def test_saved_noul_guidance_aliases_and_absent_optional_fields(self):
        request = task()
        question = request["questions"][2]
        question["true_when"] = question.pop("trueWhen")
        question["false_when"] = question.pop("falseWhen")
        output = result()
        del output["answers"]["route"]["confidence"]
        model = FakeModel(output)
        answers = evaluate(request, model, lambda: None)
        self.assertNotIn("confidence", answers[0])
        self.assertEqual(
            model.calls[0][1]["urgent"]["criteria"]["true"], "Needs action now"
        )
        del question["true_when"]
        del question["false_when"]
        model = FakeModel()
        evaluate(request, model, lambda: None)
        self.assertNotIn("criteria", model.calls[0][1]["urgent"])

    def test_rejects_unsupported_or_invalid_inputs_before_model(self):
        cases = []
        for state in (" ", "x" * 8001, False):
            request = task()
            request["state"] = state
            cases.append(request)
        for questions in ([], task()["questions"] * 3):
            request = task()
            request["questions"] = questions
            cases.append(request)
        for levels in (["one"], ["x"] * 11, ["one", " "]):
            request = task()
            request["questions"][1]["levels"] = levels
            cases.append(request)
        for field in ("key", "instructions"):
            request = task()
            request["questions"][0][field] = " "
            cases.append(request)
        request = task()
        request["questions"][1]["key"] = "route"
        cases.append(request)
        request = task()
        request["questions"][0]["options"][1]["key"] = "billing"
        cases.append(request)
        request = task()
        request["questions"][0]["options"][0]["key"] = " "
        cases.append(request)
        request = task()
        request["questions"][0]["instructions"] = "x" * 16000
        cases.append(request)
        for request in cases:
            with self.subTest(request=request):
                model = FakeModel()
                with self.assertRaises((ValueError, TypeError)):
                    evaluate(request, model, lambda: None)
                self.assertEqual(model.calls, [])

    def test_rejects_missing_extra_wrong_type_and_unknown_choice_answers(self):
        outputs = []
        output = result()
        del output["answers"]["urgent"]
        outputs.append(output)
        output = result()
        output["answers"]["extra"] = {"type": "noul", "noul": 0.5}
        outputs.append(output)
        output = result()
        output["answers"]["urgent"]["type"] = "score"
        outputs.append(output)
        output = result()
        output["answers"]["route"]["choice"] = "unknown"
        outputs.append(output)
        for output in outputs:
            with (
                self.subTest(output=output),
                self.assertRaises((ValueError, TypeError)),
            ):
                evaluate(task(), FakeModel(output), lambda: None)

    def test_rejects_invalid_numbers_and_distributions(self):
        for question, field in (
            ("priority", "score"),
            ("urgent", "noul"),
            ("route", "confidence"),
        ):
            for value in (True, float("nan"), float("inf"), -0.1, "0.5"):
                with self.subTest(question=question, field=field, value=value):
                    output = result()
                    output["answers"][question][field] = value
                    with self.assertRaises((ValueError, TypeError)):
                        evaluate(task(), FakeModel(output), lambda: None)
        for probabilities in (
            None,
            {"billing": 0.5},
            {"billing": 0.5, "wrong": 0.5},
            {"billing": 0.1, "shipping": 0.1},
            {"billing": True, "shipping": 0},
            {"billing": float("nan"), "shipping": 1},
            {"billing": -0.1, "shipping": 1.1},
        ):
            with self.subTest(probabilities=probabilities):
                output = result()
                output["answers"]["route"]["probabilities"] = probabilities
                with self.assertRaises((ValueError, TypeError)):
                    evaluate(task(), FakeModel(output), lambda: None)

        for question, field in (("urgent", "noul"), ("route", "confidence")):
            output = result()
            output["answers"][question][field] = 1.1
            with self.subTest(question=question), self.assertRaises(ValueError):
                evaluate(task(), FakeModel(output), lambda: None)

    def test_rejects_score_median_and_wrong_level_distribution(self):
        for value, probabilities in (
            (1, {"0": 0.2, "1": 0.5, "2": 0.3}),
            (1.1, {"low": 0.2, "medium": 0.5, "high": 0.3}),
            (3, {"0": 0.2, "1": 0.5, "2": 0.3}),
        ):
            with self.subTest(value=value, probabilities=probabilities):
                output = result()
                output["answers"]["priority"].update(
                    score=value, probabilities=probabilities
                )
                with self.assertRaises((ValueError, TypeError)):
                    evaluate(task(), FakeModel(output), lambda: None)

    def test_validates_optional_noul_distribution_without_inventing_it(self):
        output = result()
        output["answers"]["urgent"]["probabilities"] = {"true": 0.8, "false": 0.2}
        answers = evaluate(task(), FakeModel(output), lambda: None)
        self.assertEqual(answers[2]["probabilities"], {"true": 0.8, "false": 0.2})
        output["answers"]["urgent"]["probabilities"]["true"] = 0.7
        output["answers"]["urgent"]["probabilities"]["false"] = 0.3
        with self.assertRaises((ValueError, TypeError)):
            evaluate(task(), FakeModel(output), lambda: None)

    def test_cancellation_before_call_and_discards_late_output(self):
        class Cancelled(Exception):
            pass

        def cancelled():
            raise Cancelled

        model = FakeModel()
        with self.assertRaises(Cancelled):
            evaluate(task(), model, cancelled)
        self.assertEqual(model.calls, [])
        checks = iter((False, True))

        def check():
            if next(checks):
                raise Cancelled

        with self.assertRaises(Cancelled):
            evaluate(task(), model, check)
        self.assertEqual(len(model.calls), 1)

    def test_provider_error_propagates(self):
        class BrokenModel:
            def system_one(self, *args, **kwargs):
                raise RuntimeError("Provider failed")

        with self.assertRaisesRegex(RuntimeError, "Provider failed"):
            evaluate(task(), BrokenModel(), lambda: None)

    def test_provider_cannot_change_request_or_answer_validation_keys(self):
        class MutatingModel(FakeModel):
            def system_one(self, state, questions, *, score_readout):
                questions["route"]["criteria"]["invented"] = "An extra option"
                output = result()
                output["answers"]["route"].update(
                    choice="invented", probabilities={"invented": 1}
                )
                return output

        request = task()
        original = deepcopy(request)
        with self.assertRaises(ValueError):
            evaluate(request, MutatingModel(), lambda: None)
        self.assertEqual(request, original)

    def test_import_has_no_inference_dependencies(self):
        source = Path(__file__).parents[1] / "decision.py"
        specification = importlib.util.spec_from_file_location(
            "isolated_d1_adapter", source
        )
        module = importlib.util.module_from_spec(specification)
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            specification.loader.exec_module(module)
            answers = module.evaluate(task(), FakeModel(), lambda: None)
        self.assertTrue(callable(module.evaluate))
        self.assertEqual(len(answers), 3)


if __name__ == "__main__":
    unittest.main()
