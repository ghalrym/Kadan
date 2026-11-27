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
    ("inkling-2-mini-iq4-nl", "llm"),
    ("inkling-2-mini-q8-0", "llm"),
    ("inkling-2-mini-abliterated-iq4-nl", "llm"),
    ("inkling-2-mini-abliterated-q8-0", "llm"),
    ("lfm3-audio-2b-bf16", "audio"),
    ("qwen-tts-1.7b-custom", "speech"),
    ("whisper-medium", "transcription"),
    ("superwhisper-s1-mini", "formatting"),
    ("flux.2-klein-4b", "image"),
    ("h3-fl2va-int8-turbo", "video"),
)


def connected_worker(
    kind,
    *,
    catalog=CATALOG,
    formatting=True,
    ready=True,
    selected_model_id="inkling-2-mini-iq4-nl",
):
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
            selected_model_id=selected_model_id,
            whisper_formatting=formatting,
        ),
        ready=readiness,
        jobs=set(),
        send=AsyncMock(),
    )


class ConnectedModelOwnersTests(unittest.TestCase):
    def test_audio_worker_full_catalog_only_advertises_its_audio_model(self):
        hub = WorkerHub()
        hub.workers = {"lfm3_audio": connected_worker("lfm3_audio")}
        with patch.dict("os.environ", {}, clear=True):
            hub._refresh_models()
        self.assertEqual([model.id for model in hub.models.models], ["lfm3-audio-2b-bf16"])
        self.assertEqual(len(hub.workers["lfm3_audio"].models.models), len(CATALOG))

    def test_native_workers_advertise_only_their_fixed_owned_models(self):
        hub = WorkerHub()
        hub.workers = {kind: connected_worker(kind) for kind in NATIVE_KINDS}
        hub.primary = "llm"
        with patch.dict("os.environ", {}, clear=True):
            hub._refresh_models()
        self.assertEqual(
            {model.id for model in hub.models.models},
            {
                "inkling-2-mini-iq4-nl",
                "qwen-tts-1.7b-custom",
                "whisper-medium",
                "superwhisper-s1-mini",
                "flux.2-klein-4b",
                "h3-fl2va-int8-turbo",
            },
        )

    def test_llm_worker_q8_selection_propagates_to_snapshot_and_defaults(self):
        hub = WorkerHub()
        hub.workers = {
            "llm": connected_worker("llm", selected_model_id="inkling-2-mini-q8-0"),
            "image": connected_worker("image"),
        }
        hub.primary = "image"
        hub._refresh_models()
        self.assertEqual(hub.models.selected_model_id, "inkling-2-mini-q8-0")
        self.assertEqual(hub.models.defaults.llm, "inkling-2-mini-q8-0")
        self.assertEqual(hub.models.defaults.decisions, "inkling-2-mini-q8-0")
        self.assertEqual(
            {model.id for model in hub.models.models},
            {"inkling-2-mini-q8-0", "flux.2-klein-4b"},
        )

    def test_llm_worker_abliterated_selection_propagates_to_snapshot_and_defaults(self):
        hub = WorkerHub()
        hub.workers = {
            "llm": connected_worker(
                "llm", selected_model_id="inkling-2-mini-abliterated-iq4-nl"
            ),
            "image": connected_worker("image"),
        }
        hub.primary = "image"
        hub._refresh_models()
        self.assertEqual(
            hub.models.selected_model_id, "inkling-2-mini-abliterated-iq4-nl"
        )
        self.assertEqual(hub.models.defaults.llm, "inkling-2-mini-abliterated-iq4-nl")
        self.assertEqual(
            hub.models.defaults.decisions, "inkling-2-mini-abliterated-iq4-nl"
        )
        self.assertEqual(
            {model.id for model in hub.models.models},
            {"inkling-2-mini-abliterated-iq4-nl", "flux.2-klein-4b"},
        )

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
        audio, speech = connected_worker("lfm3_audio"), connected_worker("speech")
        computer = SimpleNamespace(
            id="computer", ledger=SimpleNamespace(workers={"speech": speech})
        )
        speech.computer = computer
        speech.controls = {}
        hub.workers = {"lfm3_audio": audio, "speech": speech}
        hub.computers = {computer.id: computer}
        hub._refresh_models()
        self.assertIn("whisper-medium", {model.id for model in hub.models.models})
        await hub._disconnected(speech)
        self.assertEqual([model.id for model in hub.models.models], ["lfm3-audio-2b-bf16"])
        self.assertNotIn(computer.id, hub.computers)

    async def test_whisper_configuration_targets_all_ready_native_stores_only(self):
        hub = WorkerHub()
        hub.workers = {
            kind: connected_worker(kind)
            for kind in (
                *NATIVE_KINDS,
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
        hub.workers = {"lfm3_audio": connected_worker("lfm3_audio")}
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
        for kind in ("lfm3_audio",):
            with self.subTest(kind=kind):
                worker = connected_worker(kind)
                await hub._synchronize(worker)
                worker.send.assert_not_awaited()
                self.assertEqual(worker.jobs, set())
                self.assertEqual(hub.jobs, {})

    async def test_native_settings_wait_for_each_persistence_confirmation(self):
        hub = WorkerHub()
        catalog = (("inkling-2-mini-iq4-nl", "llm"),)
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
