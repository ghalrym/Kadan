import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from experimental_worker.d1 import (
    CONTEXT_LIMIT,
    MODEL,
    MODEL_ID,
    REQUIREMENT,
    BoundedDecisionModel,
    execute,
)
from experimental_worker.store import ModelStore


class BoundedD1Tests(unittest.TestCase):
    def test_executor_loads_cpu_only_local_pinned_code_with_no_external_kernels(self):
        model, _ = self.model([12])
        model.eval = lambda: model
        model.parameters = lambda: [SimpleNamespace(device=SimpleNamespace(type="cpu"))]
        model.system_one.return_value = {
            "answers": {"q": {"type": "noul", "noul": 0.7}}
        }
        torch = SimpleNamespace(
            float32="f32",
            set_num_threads=MagicMock(),
            set_num_interop_threads=MagicMock(),
            inference_mode=nullcontext,
        )
        factory = MagicMock(return_value=model)
        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "transformers": SimpleNamespace(
                    AutoModel=SimpleNamespace(from_pretrained=factory)
                ),
            },
        ):
            answers = execute(
                {
                    "model_id": MODEL_ID,
                    "state": "text",
                    "questions": [
                        {"key": "q", "type": "Noul", "instructions": "True?"}
                    ],
                },
                "generate",
                "/verified",
                lambda event: None,
            )
        self.assertEqual(answers, [{"key": "q", "type": "Noul", "value": 0.7}])
        factory.assert_called_once_with(
            "/verified",
            local_files_only=True,
            trust_remote_code=True,
            dtype="f32",
            device_map="cpu",
            use_kernels=False,
        )
        torch.set_num_threads.assert_called_once_with(2)
        torch.set_num_interop_threads.assert_called_once_with(1)

    def model(self, lengths):
        engine = SimpleNamespace(
            named=lambda value: value,
            _sequences=lambda requests: [(0, 0, range(length)) for length in lengths],
        )
        model = SimpleNamespace(
            engine=engine, system_one=MagicMock(return_value={"answers": {}})
        )
        return model, BoundedDecisionModel(model)

    def test_exact_total_token_limit_precedes_forward_and_never_truncates(self):
        for lengths, allowed in (
            ([1024], True),
            ([512, 512], True),
            ([1025], False),
            ([600, 600], False),
            ([], False),
            ([0], False),
        ):
            with self.subTest(lengths=lengths):
                model, bounded = self.model(lengths)
                if allowed:
                    bounded.system_one("state", {"q": {}}, score_readout="mean")
                    model.system_one.assert_called_once_with(
                        "state", {"q": {}}, score_readout="mean"
                    )
                else:
                    with self.assertRaises(ValueError):
                        bounded.system_one("state", {"q": {}}, score_readout="mean")
                    model.system_one.assert_not_called()

    def test_invalid_task_or_operation_does_not_import_inference_libraries(self):
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            for task, operation in (
                ({}, "load"),
                ({}, "generate"),
                ({"model_id": MODEL_ID}, "generate"),
            ):
                with (
                    self.subTest(task=task, operation=operation),
                    self.assertRaises(ValueError),
                ):
                    execute(task, operation, "/unused", lambda event: None)

    def test_catalog_is_exact_pinned_official_snapshot_and_reserves_cpu_ram(self):
        manifest = json.loads(
            (Path(__file__).parents[1] / "catalog_d1.json").read_text()
        )
        store = ModelStore(Path("/unused"), MODEL, manifest)
        self.assertEqual(REQUIREMENT, (15 * 1024**3, 0))
        self.assertEqual(CONTEXT_LIMIT, 1024)
        self.assertEqual(len(store.artifacts), 34)
        self.assertEqual(sum(item["size"] for item in store.artifacts), 3_854_681_959)
        self.assertTrue(
            all(
                item["algorithm"] == "sha256" and len(item["digest"]) == 64
                for item in store.artifacts
            )
        )
        self.assertEqual(MODEL["revision"], "10e7671d44e4b37700020b1eed3f594c5f2af0b4")
