import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import (
    ChatTask,
    InferenceTask,
    TranscriptionTask,
)
from api.pydantic_models.model_store import ModelAssignments, ModelStoreStatus
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.settings import SettingsRequest, get_settings
from api.services import transcription
from api.services.inference_errors import InferenceFailure
from api.services.worker_hub import WorkerHub, target_worker


def checkpoint(model_id, kind):
    return {
        "id": model_id,
        "repo_id": "unit/model",
        "revision": "pinned",
        "license": "unit",
        "estimated_bytes": 10,
        "kind": kind,
        "display_name": model_id,
        "license_url": None,
        "license_notice": None,
        "inference_available": True,
        "status": "not_downloaded",
        "downloaded_bytes": 0,
        "total_bytes": 10,
        "error": None,
        "context_limit": None,
        "architecture_context_limit": None,
    }


class FixedStackTests(unittest.TestCase):
    def test_default_chat_selects_structured_lfm_without_feature_flags(self):
        body = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], tools=[]
        )
        task = InferenceTask(
            operation="completion",
            input={"feature": "llm", **body.model_dump(exclude_unset=True)},
        )
        self.assertEqual(task.input.model, "lfm3-1.7b-bf16")
        self.assertEqual(target_worker(task), "lfm3_llm")
        self.assertEqual(task.model_dump()["input"]["model"], "lfm3-1.7b-bf16")

    def test_removed_language_model_is_rejected_instead_of_substituted(self):
        for model in ("qwen3.6-35b-a3b-q4-k-m", "", "other"):
            with self.subTest(model=model), self.assertRaises(ValidationError):
                ChatTask(model=model, messages=[{"role": "user", "text": "Hello"}])

    def test_tier_settings_are_no_longer_accepted(self):
        with self.assertRaises(ValidationError):
            SettingsRequest(tier="go", license_acknowledged=True)
        with self.assertRaises(ValidationError):
            InferenceTask(
                operation="configure", input={"feature": "tier_selection", "tier": "go"}
            )

    def test_fixed_assignments_do_not_fabricate_connected_checkpoints(self):
        hub = WorkerHub()
        native = ModelStoreStatus(
            models=[
                checkpoint("qwen-image-2.1-turbo", "image"),
                checkpoint("qwen-tts-1.7b-custom", "speech"),
            ]
        )
        hub.primary = "image"
        hub.workers = {"image": SimpleNamespace(models=native, kind="image")}
        hub._refresh_models()
        self.assertEqual(
            [row.id for row in hub.models.models], ["qwen-image-2.1-turbo"]
        )
        self.assertEqual(hub.models.selected_model_id, "lfm3-1.7b-bf16")
        self.assertEqual(hub.models.defaults.vision, "lfm3-vl-2b-bf16")
        with patch(
            "api.routes.v1.settings.inference.models_status", return_value=hub.models
        ):
            result = get_settings()
        self.assertEqual(len(result.models), 1)
        self.assertNotIn("tiers", result.model_dump())
        self.assertEqual(
            (result.video_defaults.duration, result.video_defaults.resolution),
            (5, "480p"),
        )

    def test_old_language_download_cannot_reach_any_worker(self):
        hub = WorkerHub()
        task = InferenceTask(
            operation="download",
            input={"feature": "models", "model_id": "qwen3.6-35b-a3b-q4-k-m"},
        )
        with self.assertRaisesRegex(InferenceFailure, "fixed stack"):
            hub._store_targets(task)

    def test_native_catalog_requires_a_connected_owner_for_each_kind(self):
        hub = WorkerHub()
        catalog = ModelStoreStatus(
            models=[
                checkpoint("laya", "decision"),
                checkpoint("qwen-image-2.1-turbo", "image"),
                checkpoint("whisper-medium", "transcription"),
                checkpoint("h3-fl2va-int8-turbo", "video"),
            ],
            whisper_formatting=False,
        )
        hub.primary = "decision"
        hub.workers = {"decision": SimpleNamespace(models=catalog, kind="decision")}
        hub._refresh_models()
        self.assertEqual([model.id for model in hub.models.models], ["laya"])
        self.assertFalse(hub.models.whisper_formatting)
        hub.workers["image"] = SimpleNamespace(models=catalog, kind="image")
        hub._refresh_models()
        self.assertEqual(
            [model.id for model in hub.models.models], ["laya", "qwen-image-2.1-turbo"]
        )
        hub.workers.pop("image")
        hub._refresh_models()
        self.assertEqual([model.id for model in hub.models.models], ["laya"])

    def test_transcription_settings_only_target_native_persistence(self):
        hub = WorkerHub()
        language = SimpleNamespace(kind="lfm3_llm")
        image = SimpleNamespace(kind="image")
        task = InferenceTask(
            operation="configure",
            input={"feature": "whisper_formatting", "enabled": False},
        )
        with patch.object(hub, "_ready", return_value=[language, image]):
            self.assertEqual(hub._store_targets(task), [image])
        with patch.object(hub, "_ready", return_value=[language]):
            with self.assertRaises(InferenceFailure) as error:
                hub._store_targets(task)
        self.assertEqual(error.exception.status_code, 503)

    def test_fixed_medium_selection_reaches_native_store_but_other_sizes_do_not(self):
        hub = WorkerHub()
        speech = SimpleNamespace(kind="speech")
        with patch.object(hub, "_ready", return_value=[speech]):
            task = InferenceTask(
                operation="configure",
                input={"feature": "whisper_selection", "model": "medium"},
            )
            self.assertEqual(hub._store_targets(task), [speech])
            other = InferenceTask(
                operation="configure",
                input={"feature": "whisper_selection", "model": "tiny.en"},
            )
            with self.assertRaises(InferenceFailure) as error:
                hub._store_targets(other)
        self.assertEqual(error.exception.status_code, 422)


class LfmFormattingTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_replica_receives_context_from_observed_lfm_catalog(self):
        hub = WorkerHub()
        source = checkpoint("lfm3-1.7b-bf16", "llm")
        source["context_limit"] = 4096
        target = {**source, "context_limit": 8192}
        hub.models = ModelStoreStatus(models=[source])
        worker = SimpleNamespace(
            id="replica",
            kind="lfm3_llm",
            models=ModelStoreStatus(models=[target]),
            jobs=set(),
        )
        sent = []

        async def send(message):
            sent.append(message)
            hub.jobs[message["id"]].result.set_result({})

        worker.send = send
        await hub._synchronize(worker)
        self.assertEqual(
            [message["input"] for message in sent],
            [
                {
                    "feature": "model_context",
                    "model_id": "lfm3-1.7b-bf16",
                    "context_limit": 4096,
                }
            ],
        )
        self.assertFalse(hub.jobs)
        self.assertFalse(worker.jobs)

    async def test_lfm_replica_does_not_acknowledge_unowned_native_preferences(self):
        hub = WorkerHub()
        model = checkpoint("lfm3-1.7b-bf16", "llm")
        hub.models = ModelStoreStatus(
            models=[model], selected_whisper="medium", whisper_formatting=False
        )
        worker = SimpleNamespace(
            kind="lfm3_llm", models=ModelStoreStatus(models=[model]), send=AsyncMock()
        )
        await hub._synchronize(worker)
        worker.send.assert_not_awaited()

    def raw(self):
        return {
            "text": "hello there",
            "raw_text": "hello there",
            "language": "en",
            "model": "medium",
            "formatting_status": "requested",
            "formatting_model": None,
        }

    async def test_formatter_uses_fixed_lfm_queue_and_preserves_whisper_raw(self):
        raw = self.raw()
        with patch.object(
            transcription.inference,
            "submit",
            new=AsyncMock(
                side_effect=[
                    raw,
                    {
                        "text": "Hello there.",
                        "model": "lfm3-1.7b-bf16",
                        "finish_reason": "stop",
                    },
                ]
            ),
        ) as submit:
            result = await transcription.transcribe(TranscriptionTask(audio="unit"))
        task = submit.await_args_list[1].args[0]
        self.assertEqual(task.model, "lfm3-1.7b-bf16")
        self.assertEqual(task.messages[-1].text, raw["raw_text"])
        self.assertEqual(
            submit.await_args_list[1].kwargs,
            {"feature": "llm", "operation": "completion"},
        )
        self.assertEqual(result["raw_text"], raw["raw_text"])
        self.assertEqual(result["model"], "medium")
        self.assertEqual(result["formatting_model"], "lfm3-1.7b-bf16")

    async def test_incomplete_or_command_output_retains_raw_with_warning(self):
        for text, finish in [
            ("", "stop"),
            ("Partial", "length"),
            ("<tool_call>bad", "stop"),
        ]:
            raw = self.raw()
            with (
                self.subTest(text=text),
                patch.object(
                    transcription.inference,
                    "submit",
                    new=AsyncMock(
                        side_effect=[
                            raw,
                            {
                                "text": text,
                                "model": "lfm3-1.7b-bf16",
                                "finish_reason": finish,
                            },
                        ]
                    ),
                ),
            ):
                result = await transcription.transcribe(TranscriptionTask(audio="unit"))
            self.assertEqual(result["text"], raw["raw_text"])
            self.assertEqual(result["formatting_status"], "failed")
            self.assertTrue(result["formatting_error"])
