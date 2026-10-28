"""Pure d1 adapter and metadata units; never load a tokenizer or model."""

import copy
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experimental_worker.d1 import (
    MAX_CONTEXT,
    MODEL,
    InputError,
    answers,
    evaluate,
    execute,
    prepare_questions,
)


def task():
    return {
        "model": MODEL["id"],
        "feature": "decisions",
        "state": "A support ticket.",
        "questions": [
            {
                "key": "team",
                "type": "Choice",
                "instructions": "Route this ticket.",
                "options": [
                    {"key": "z", "description": "Billing"},
                    {"key": "a", "description": "Shipping"},
                ],
            },
            {
                "key": "urgency",
                "type": "Score",
                "instructions": "How urgent?",
                "levels": ["low", "medium", "high"],
            },
            {
                "key": "refund",
                "type": "Noul",
                "instructions": "Refund requested?",
                "true_when": "A refund is requested.",
                "false_when": "No refund.",
            },
        ],
    }


def result():
    # Deliberately different mapping order from the request.
    return {
        "answers": {
            "refund": {"type": "noul", "noul": 0.2},
            "urgency": {
                "type": "score",
                "score": 1.1,
                "confidence": 0.5,
                "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
            },
            "team": {
                "type": "choice",
                "choice": "z",
                "confidence": 0.75,
                "probabilities": {"a": 0.25, "z": 0.75},
            },
        }
    }


class D1AdapterTests(unittest.TestCase):
    def test_maps_question_names_option_order_rubric_order_and_binary_guidance(self):
        original = task()
        before = copy.deepcopy(original)
        named = prepare_questions(original)
        self.assertEqual(list(named), ["team", "urgency", "refund"])
        self.assertEqual(list(named["team"]["criteria"]), ["z", "a"])
        self.assertEqual(named["team"]["criteria"]["z"], "Billing")
        self.assertEqual(named["urgency"]["criteria"], ["low", "medium", "high"])
        self.assertEqual(
            named["refund"]["criteria"],
            {
                "true": "A refund is requested.",
                "false": "No refund.",
            },
        )
        self.assertEqual(original, before)

    def test_http_binary_aliases_and_empty_guidance_have_explicit_meaning(self):
        request = task()
        request["questions"] = [
            {
                "key": "x",
                "type": "Noul",
                "instructions": "True?",
                "trueWhen": "Yes",
                "falseWhen": "",
            }
        ]
        self.assertEqual(
            prepare_questions(request)["x"]["criteria"],
            {
                "true": "Yes",
                "false": "no, the statement does not hold",
            },
        )
        request["questions"][0]["true_when"] = "Conflicting alias"
        with self.assertRaises(InputError):
            prepare_questions(request)

    def test_restores_request_order_and_preserves_fractional_mean_and_labels(self):
        output = answers(task(), result())
        self.assertEqual([row["key"] for row in output], ["team", "urgency", "refund"])
        self.assertEqual(output[0]["value"], "z")
        self.assertEqual(list(output[0]["probabilities"]), ["z", "a"])
        self.assertEqual(output[1]["value"], 1.1)
        self.assertEqual(output[1]["probabilities"], {"0": 0.2, "1": 0.5, "2": 0.3})
        self.assertEqual(output[2]["value"], 0.2)
        self.assertIsNone(output[2]["confidence"])
        self.assertIsNone(output[2]["probabilities"])

    def test_invalid_outputs_are_errors_not_fabricated_decisions(self):
        cases = [
            ("missing", lambda value: value["answers"].pop("refund")),
            (
                "unknown",
                lambda value: value["answers"]["team"].update(choice="missing"),
            ),
            (
                "nonmax",
                lambda value: value["answers"]["team"].update(
                    choice="a", confidence=0.25
                ),
            ),
            ("median", lambda value: value["answers"]["urgency"].update(score=1)),
            (
                "nonfinite",
                lambda value: value["answers"]["refund"].update(noul=float("nan")),
            ),
            ("boolean", lambda value: value["answers"]["refund"].update(noul=True)),
            (
                "confidence",
                lambda value: value["answers"]["team"].update(confidence=0.5),
            ),
            (
                "distribution",
                lambda value: value["answers"]["team"].update(
                    probabilities={"z": 0.3, "a": 0.3}
                ),
            ),
            (
                "labels",
                lambda value: value["answers"]["urgency"].update(
                    probabilities={"low": 0.2, "medium": 0.5, "high": 0.3}
                ),
            ),
        ]
        for name, change in cases:
            with self.subTest(name=name):
                value = result()
                change(value)
                with self.assertRaises(ValueError):
                    answers(task(), value)

    def test_unsupported_tasks_and_score_ranges_are_rejected_before_model_import(self):
        for levels in (["one"], [str(index) for index in range(11)]):
            request = task()
            request["questions"][1]["levels"] = levels
            with self.assertRaisesRegex(InputError, "2 to 10"):
                execute(request, "generate", "/not-loaded", lambda event: None)
        for changes in ({"media": ["image"]}, {"feature": "llm"}, {"model": "laya"}):
            with self.assertRaises(InputError):
                prepare_questions({**task(), **changes})
        request = task()
        request["questions"][0]["multiple"] = True
        with self.assertRaises(InputError):
            prepare_questions(request)
        with self.assertRaises(InputError):
            execute(task(), "completion", "/not-loaded", lambda event: None)

    def test_full_prompt_token_limit_includes_questions_and_uses_mean_readout(self):
        request = task()
        request["state"] = "x" * 20000
        for count in (128000, 128001, 0):
            with self.subTest(tokens=count):
                engine = SimpleNamespace(
                    ask=Mock(side_effect=lambda question: question),
                    tokens=Mock(return_value=count),
                )
                model = SimpleNamespace(
                    engine=engine, system_one=Mock(return_value=result())
                )
                if count == MAX_CONTEXT:
                    self.assertEqual(evaluate(model, request)[1]["value"], 1.1)
                    model.system_one.assert_called_once_with(
                        request["state"],
                        prepare_questions(request),
                        score_readout="mean",
                        set_size=1,
                    )
                else:
                    with self.assertRaisesRegex(InputError, "no input was truncated"):
                        evaluate(model, request)
                    model.system_one.assert_not_called()
                engine.tokens.assert_called_once_with(
                    request["state"], list(prepare_questions(request).values())
                )

    def test_snapshot_pins_weights_custom_code_calibration_and_license(self):
        self.assertEqual(MODEL["repo_id"], "LiquidAI/d1-2B")
        self.assertEqual(MODEL["revision"], "cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41")
        self.assertEqual(MODEL["context_limit"], 128000)
        self.assertEqual(MODEL["license"], "lfm1.0")
        files = {item["name"]: item for item in MODEL["artifacts"]}
        for name in (
            "LICENSE",
            "model.safetensors",
            "modeling_d1.py",
            "runner.py",
            "prompt.py",
            "calibration.json",
            "tokenizer.json",
            "native_lfm3_vl.py",
        ):
            self.assertIn(name, files)
        self.assertEqual(
            sum(item["size"] for item in files.values()), MODEL["estimated_bytes"]
        )
        self.assertEqual(files["model.safetensors"]["algorithm"], "sha256")
        for item in files.values():
            self.assertIn(f"/resolve/{MODEL['revision']}/", item["url"])

    def test_execution_loads_only_the_local_bf16_snapshot_and_reads_decisions(self):
        model = Mock()
        model.eval.return_value = model
        model.engine.ask.side_effect = lambda question: question
        model.engine.tokens.return_value = 100
        model.system_one.return_value = result()
        auto_model = SimpleNamespace(from_pretrained=Mock(return_value=model))
        torch = SimpleNamespace(
            bfloat16=object(),
            inference_mode=nullcontext,
            cuda=SimpleNamespace(
                is_available=lambda: True, is_bf16_supported=lambda: True
            ),
        )
        events = []
        with patch.dict(
            "sys.modules",
            {"torch": torch, "transformers": SimpleNamespace(AutoModel=auto_model)},
        ):
            response = execute(task(), "generate", "/verified/local/d1", events.append)
        auto_model.from_pretrained.assert_called_once_with(
            "/verified/local/d1",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.assertEqual(response["model"], MODEL["id"])
        self.assertEqual(response["answers"][1]["value"], 1.1)
        self.assertEqual(events, [{"state": "running", "model": MODEL["id"]}])
