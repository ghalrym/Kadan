import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_errors import InferenceFailure
from api.services.worker_hub import WorkerHub
from api.tests.services.test_worker_hub import checkpoint

NATIVE_KINDS = ("llm", "tts", "speech", "formatting", "image", "video")
CATALOG = (
    ("lfm3-1.7b-bf16", "llm"),
    ("lfm3-vl-2b-bf16", "vision"),
    ("lfm3-audio-2b-bf16", "audio"),
    ("d1-2b-cpu-f32", "decision"),
    ("qwen-tts-1.7b-custom", "speech"),
    ("whisper-medium", "transcription"),
    ("superwhisper-s1-mini", "formatting"),
    ("qwen-image-2.1-turbo", "image"),
    ("h3-fl2va-int8-turbo", "video"),
    ("qwen3.6-35b-a3b-q4-k-m", "llm"),
)


def connected_worker(kind, *, catalog=CATALOG, formatting=True, ready=True):
    readiness = asyncio.Event()
    if ready:
        readiness.set()
    return SimpleNamespace(
        id=kind,
        kind=kind,
        name=kind,
        models=ModelStoreStatus(
            models=[
                checkpoint(model_id, model_kind) for model_id, model_kind in catalog
            ],
            whisper_formatting=formatting,
        ),
        ready=readiness,
        jobs=set(),
        send=AsyncMock(),
    )


class ConnectedModelOwnersTests(unittest.TestCase):
    def test_decision_worker_full_catalog_only_advertises_its_decision_model(self):
        hub = WorkerHub()
        hub.workers = {"lfm3_decision": connected_worker("lfm3_decision")}
        with patch.dict("os.environ", {}, clear=True):
            hub._refresh_models()
        self.assertEqual([model.id for model in hub.models.models], ["d1-2b-cpu-f32"])
        self.assertEqual(len(hub.workers["lfm3_decision"].models.models), len(CATALOG))

    def test_native_workers_advertise_only_their_fixed_owned_models(self):
        hub = WorkerHub()
        hub.workers = {kind: connected_worker(kind) for kind in NATIVE_KINDS}
        hub.primary = "llm"
        with patch.dict("os.environ", {}, clear=True):
            hub._refresh_models()
        self.assertEqual(
            {model.id for model in hub.models.models},
            {
                "qwen-tts-1.7b-custom",
                "whisper-medium",
                "superwhisper-s1-mini",
                "qwen-image-2.1-turbo",
                "h3-fl2va-int8-turbo",
            },
        )

    def test_text_and_vision_flags_do_not_disable_default_audio(self):
        hub = WorkerHub()
        hub.workers = {
            kind: connected_worker(kind)
            for kind in ("lfm3_decision", "lfm3_llm", "lfm3_vision", "lfm3_audio")
        }
        for text_enabled, vision_enabled in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with (
                self.subTest(text=text_enabled, vision=vision_enabled),
                patch.dict(
                    "os.environ",
                    {
                        "HIKARI_ENABLE_LFM3_TEXT": str(text_enabled).lower(),
                        "HIKARI_ENABLE_LFM3_VISION": str(vision_enabled).lower(),
                    },
                    clear=True,
                ),
            ):
                hub._refresh_models()
                expected = {"d1-2b-cpu-f32", "lfm3-audio-2b-bf16"}
                if text_enabled:
                    expected.add("lfm3-1.7b-bf16")
                if vision_enabled:
                    expected.add("lfm3-vl-2b-bf16")
                self.assertEqual({model.id for model in hub.models.models}, expected)

    def test_lfm_only_snapshot_includes_default_audio_without_native_primary(self):
        hub = WorkerHub()
        hub.workers = {"lfm3_audio": connected_worker("lfm3_audio")}
        with patch.dict("os.environ", {}, clear=True):
            hub._refresh_models()
        self.assertEqual(
            [model.id for model in hub.models.models], ["lfm3-audio-2b-bf16"]
        )
        self.assertEqual(hub.models.defaults.transcription, "whisper-medium")

    def test_primary_saved_formatting_is_preserved_when_other_workers_disagree(self):
        hub = WorkerHub()
        hub.workers = {
            "tts": connected_worker("tts", formatting=False),
            "speech": connected_worker("speech", formatting=True),
            "lfm3_audio": connected_worker("lfm3_audio", formatting=True),
        }
        hub.primary = "tts"
        hub._refresh_models()
        self.assertFalse(hub.models.whisper_formatting)
        self.assertFalse(hub.workers["tts"].models.whisper_formatting)
        self.assertTrue(hub.workers["speech"].models.whisper_formatting)

    def test_snapshot_rows_are_deep_copied_and_duplicate_owners_are_deduplicated(self):
        first = connected_worker(
            "speech", catalog=(("whisper-medium", "transcription"),)
        )
        second = connected_worker(
            "speech", catalog=(("whisper-medium", "transcription"),)
        )
        first.models.models.append(first.models.models[0].model_copy(deep=True))
        hub = WorkerHub()
        hub.primary = "first"
        hub.workers = {"first": first, "second": second}
        hub._refresh_models()
        self.assertEqual(len(hub.models.models), 1)
        hub.models.models[0].status = "failed"
        hub.models.models[0].error = "snapshot only"
        hub.models.video_defaults.duration = 10
        for worker in (first, second):
            self.assertTrue(
                all(model.status == "complete" for model in worker.models.models)
            )
            self.assertTrue(all(model.error is None for model in worker.models.models))
            self.assertEqual(worker.models.video_defaults.duration, 5)


class ConnectedStoreSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnected_owner_is_not_replaced_by_another_native_catalog(self):
        hub = WorkerHub()
        decision, speech = connected_worker("lfm3_decision"), connected_worker("speech")
        computer = SimpleNamespace(
            id="computer", ledger=SimpleNamespace(workers={"speech": speech})
        )
        speech.computer = computer
        speech.controls = {}
        hub.workers = {"lfm3_decision": decision, "speech": speech}
        hub.computers = {computer.id: computer}
        hub._refresh_models()
        self.assertIn("whisper-medium", {model.id for model in hub.models.models})
        await hub._disconnected(speech)
        self.assertEqual([model.id for model in hub.models.models], ["d1-2b-cpu-f32"])
        self.assertNotIn(computer.id, hub.computers)

    async def test_whisper_configuration_targets_all_ready_native_stores_only(self):
        hub = WorkerHub()
        hub.workers = {
            kind: connected_worker(kind)
            for kind in (
                *NATIVE_KINDS,
                "lfm3_decision",
                "lfm3_llm",
                "lfm3_vision",
                "lfm3_audio",
            )
        }
        hub.workers["not-ready"] = connected_worker("speech", ready=False)
        request = InferenceTask(
            operation="configure",
            input={"feature": "whisper_formatting", "enabled": False},
        )
        self.assertEqual(
            {worker.kind for worker in hub._store_targets(request)}, set(NATIVE_KINDS)
        )
        self.assertNotIn(hub.workers["not-ready"], hub._store_targets(request))

    async def test_whisper_configuration_without_native_owner_reports_unavailable(self):
        hub = WorkerHub()
        hub.workers = {
            kind: connected_worker(kind)
            for kind in ("lfm3_decision", "lfm3_llm", "lfm3_vision", "lfm3_audio")
        }
        request = InferenceTask(
            operation="configure",
            input={"feature": "whisper_formatting", "enabled": False},
        )
        with self.assertRaises(InferenceFailure) as raised:
            hub._store_targets(request)
        self.assertEqual(raised.exception.status_code, 503)

    async def test_lfm_workers_receive_no_native_settings_on_connection(self):
        hub = WorkerHub()
        primary = connected_worker("llm", formatting=False)
        primary.models.models[0].context_limit = 8192
        hub.primary = "llm"
        hub.workers = {"llm": primary}
        for kind in ("lfm3_decision", "lfm3_llm", "lfm3_vision", "lfm3_audio"):
            with self.subTest(kind=kind):
                worker = connected_worker(kind)
                await hub._synchronize(worker)
                worker.send.assert_not_awaited()
                self.assertEqual(worker.jobs, set())
                self.assertEqual(hub.jobs, {})

    async def test_native_settings_wait_for_each_persistence_confirmation(self):
        hub = WorkerHub()
        catalog = (("qwen3.6-35b-a3b-q4-k-m", "llm"),)
        primary = connected_worker("llm", catalog=catalog, formatting=False)
        primary.models.models[0].context_limit = 8192
        hub.primary = "primary"
        hub.workers = {"primary": primary}
        for kind in NATIVE_KINDS:
            with self.subTest(kind=kind):
                worker = connected_worker(kind, catalog=catalog)
                submissions = asyncio.Queue()

                async def send(message, submissions=submissions):
                    await submissions.put(message)

                worker.send.side_effect = send
                synchronize = asyncio.create_task(hub._synchronize(worker))
                try:
                    for expected in (
                        {"feature": "whisper_formatting", "enabled": False},
                        {
                            "feature": "model_context",
                            "model_id": catalog[0][0],
                            "context_limit": 8192,
                        },
                    ):
                        message = await asyncio.wait_for(submissions.get(), 1)
                        self.assertEqual(message["operation"], "configure")
                        self.assertEqual(message["input"], expected)
                        self.assertFalse(synchronize.done())
                        self.assertIn(message["id"], worker.jobs)
                        hub.jobs[message["id"]].result.set_result({})
                    await asyncio.wait_for(synchronize, 1)
                finally:
                    if not synchronize.done():
                        synchronize.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await synchronize
                self.assertEqual(worker.jobs, set())
                self.assertEqual(hub.jobs, {})
