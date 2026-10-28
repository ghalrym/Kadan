import asyncio
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import InferenceJob, JobState, WorkerEvent
from api.services.memory_scheduler import GIB, MachineLedger, Requirement, WorkerSlot
from api.services.worker_hub import (
    RELEASE_TIMEOUT,
    Computer,
    Worker,
    WorkerHub,
    WorkerSettings,
    authorized,
    requested_model,
)


class FakeSocket:
    """One worker container's connection, seen from the API."""

    def __init__(self, token="secret", replies=None):
        self.headers = {"authorization": f"Bearer {token}"}
        self.inbox = asyncio.Queue()
        self.sent = []
        self.closed = None
        self.replies = replies or {}

    async def receive_text(self):
        item = await self.inbox.get()
        if item is None:
            raise WebSocketDisconnect()
        return item

    async def send_text(self, text):
        message = json.loads(text)
        self.sent.append(message)
        reply = self.replies.get(message.get("command"))
        if reply is not None:
            for response in reply(message):
                self.push(response)

    def push(self, message):
        self.inbox.put_nowait(json.dumps(message))

    async def close(self, code=1000):
        self.closed = code
        self.inbox.put_nowait(None)

    def commands(self, command):
        return [message for message in self.sent if message.get("command") == command]


def hello(kind, *, computer="boot-1", instance=None, gpus=(("GPU-0", 24),), protocol=4):
    return {
        "state": "hello",
        "protocol": protocol,
        "pid": 1,
        "worker": kind,
        "instance": instance or kind + "-1",
        "machine": {
            "id": computer,
            "name": "box-" + computer,
            "ram_total": 68 * GIB,
            "ram_free": 64 * GIB,
            "devices": [
                {
                    "index": index,
                    "uuid": uuid,
                    "vram_total": 25 * GIB,
                    "vram_free": free * GIB,
                }
                for index, (uuid, free) in enumerate(gpus)
            ],
        },
    }


def models(formatting=True, whisper=None, checkpoints=()):
    return {
        "models": list(checkpoints),
        "selected_model_id": "lfm3-1.7b-bf16",
        "selected_whisper": whisper,
        "whisper_formatting": formatting,
    }


def expired():
    """A release start older than the timeout on this machine's clock."""
    return time.monotonic() - RELEASE_TIMEOUT - 1


def queued(message):
    return [{"id": message["id"], "state": "queued"}]


def configured(message):
    """A store lane that applies settings at once."""
    return [
        {"id": message["id"], "state": "queued"},
        {"id": message["id"], "state": "succeeded", "result": {}},
    ]


def ok(message):
    return [{"id": message["id"], "state": "control", "ok": True}]


def busy(message):
    return [{"id": message["id"], "state": "control", "ok": False, "error": "busy"}]


def needs(minimum, preferred, floor=0, model="m"):
    """A requirement reply in GiB."""

    def reply(message):
        return [
            {
                "id": message["id"],
                "state": "requirement",
                "model": model,
                "minimum": [value * GIB for value in minimum],
                "preferred": [value * GIB for value in preferred],
                "device_floor": floor * GIB,
            }
        ]

    return reply


def checkpoint(model_id, kind="llm", context_limit=None):
    return {
        "status": "complete",
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "error": None,
        "id": model_id,
        "repo_id": "org/" + model_id,
        "revision": "r",
        "license": "apache-2.0",
        "estimated_bytes": GIB,
        "kind": kind,
        "display_name": None,
        "license_url": None,
        "license_notice": None,
        "inference_available": True,
        "context_limit": context_limit,
        "architecture_context_limit": 262144,
    }


def chat():
    return InferenceTask(
        operation="completion",
        input={"feature": "llm", "messages": [{"role": "user", "text": "hi"}]},
    )


def image():
    return InferenceTask(input={"feature": "image", "prompt": "fox", "count": 1})


class HubTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.timeout = patch("api.services.worker_hub.CONNECT_TIMEOUT", 0.05)
        self.timeout.start()
        self.addCleanup(self.timeout.stop)
        self.events = []
        self.hub = WorkerHub(
            on_event=lambda job, event: self.events.append((job, event))
        )
        self.tasks = []

    async def asyncTearDown(self):
        for socket in [task.socket for task in self.tasks]:
            await socket.close()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        # Disconnecting fails outstanding jobs; nothing else awaits them here.
        for job in self.hub.jobs.values():
            for future in (job.accepted, job.result):
                if future.done() and not future.cancelled():
                    future.exception()

    async def connect(self, kind, *, store=None, replies=None, **identity):
        socket = FakeSocket(replies={"submit": queued, **(replies or {})})
        socket.push(hello(kind, **identity))
        socket.push({"state": "models", "models": store or models()})
        socket.push({"state": "ready", "protocol": 4, "pid": 1})
        known = set(self.hub.workers)
        task = asyncio.create_task(self.hub.serve(socket))
        task.socket = socket
        self.tasks.append(task)
        for _ in range(100):
            for worker_id, worker in self.hub.workers.items():
                if worker_id not in known and worker.ready.is_set():
                    return socket, worker
            await asyncio.sleep(0)
        self.fail(f"{kind} did not connect")

    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0)


class AuthorizationTests(unittest.TestCase):
    def test_token_must_match(self):
        socket = SimpleNamespace(headers={"authorization": "Bearer secret"})
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(authorized(socket))
        with patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": "other"}):
            self.assertFalse(authorized(socket))
        with patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": "secret"}):
            self.assertTrue(authorized(socket))
            self.assertFalse(authorized(SimpleNamespace(headers={})))

    def test_default_api_key_is_used_when_unset_or_empty(self):
        socket = SimpleNamespace(headers={"authorization": "Bearer insecure-api-key"})
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(authorized(socket))
            self.assertFalse(authorized(SimpleNamespace(headers={})))
        with patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": ""}):
            self.assertTrue(authorized(socket))

    def test_requested_model_reads_either_field(self):
        video = InferenceTask(input={"feature": "video", "prompt": "sea"})
        self.assertEqual(requested_model(video), "h3-fl2va-int8-turbo")
        self.assertEqual(requested_model(chat()), "lfm3-1.7b-bf16")


class RegistrationTests(HubTestCase):
    async def test_unsupported_protocol_and_unknown_types_are_rejected(self):
        for message in (hello("lfm3_llm", protocol=3), hello("gpu")):
            socket = FakeSocket()
            socket.push(message)
            await self.hub.serve(socket)
            self.assertEqual(socket.closed, 1008)
        self.assertFalse(self.hub.workers)

    async def test_workers_on_one_computer_share_its_ram_and_each_gpu_once(self):
        _, language = await self.connect("lfm3_llm", gpus=(("GPU-a", 24),))
        _, picture = await self.connect("image", gpus=(("GPU-a", 24),))
        self.assertIs(language.computer, picture.computer)
        ledger = language.computer.ledger
        self.assertEqual(ledger.capacity, [64 * GIB, 24 * GIB])
        self.assertEqual(set(ledger.workers), {language.id, picture.id})

    async def test_workers_on_other_computers_have_their_own_ledgers(self):
        _, here = await self.connect("lfm3_llm", computer="boot-1")
        _, there = await self.connect("lfm3_llm", computer="boot-2", instance="llm-2")
        self.assertIsNot(here.computer, there.computer)
        self.assertEqual(len(self.hub.computers), 2)

    async def test_a_worker_that_sees_some_gpus_gets_budgets_in_its_own_order(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        # The language worker sees both GPUs and will start on GPU-a.
        await self.connect(
            "lfm3_llm",
            gpus=(("GPU-a", 24), ("GPU-b", 20)),
            replies={"requirement": needs((1, 6), (1, 6))},
        )
        _, picture = await self.connect(
            "image",
            gpus=(("GPU-b", 20),),
            replies={"requirement": needs((1, 2), (1, 8)), "load": load},
        )
        self.assertEqual(picture.devices, [2])
        await self.hub.submit(image())
        await self.hub._schedule(picture.computer)
        self.assertEqual(loads, [[GIB, 8 * GIB]])
        self.assertEqual(picture.slot.budget, [GIB, 0, 8 * GIB])

    async def test_cache_idle_setting_is_sent_with_worker_budget(self):
        self.hub.kv_cache_idle_seconds = 45
        socket, worker = await self.connect(
            "lfm3_llm", replies={"load": ok, "requirement": needs((1, 6), (1, 6))}
        )
        await self.hub.submit(chat())
        await self.hub._schedule(worker.computer)
        self.assertEqual(socket.commands("load")[-1]["kv_cache_idle_seconds"], 45)

    async def test_computer_ledger_goes_when_its_last_worker_leaves(self):
        socket, worker = await self.connect("lfm3_llm")
        await socket.close()
        await self.settle()
        self.assertNotIn(worker.computer.id, self.hub.computers)

    async def test_first_worker_is_primary_for_model_status(self):
        await self.connect(
            "image",
            store=models(checkpoints=[checkpoint("qwen-image-2.1-turbo", "image")]),
        )
        self.assertEqual(self.hub.models.selected_model_id, "lfm3-1.7b-bf16")

    async def test_late_worker_receives_saved_whisper_preferences_without_a_tier(self):
        primary = models(formatting=False, whisper="medium")
        await self.connect("image", store=primary)
        socket, _ = await self.connect(
            "video", computer="boot-2", store=models(), replies={"submit": configured}
        )
        sent = [message["input"] for message in socket.commands("submit")]
        self.assertIn({"feature": "whisper_selection", "model": "medium"}, sent)
        self.assertIn({"feature": "whisper_formatting", "enabled": False}, sent)
        self.assertNotIn("tier_selection", [message["feature"] for message in sent])

    async def test_late_worker_is_not_routable_until_settings_apply(self):
        await self.connect("speech", store=models(formatting=False))
        socket = FakeSocket(
            replies={
                "submit": lambda m: [
                    {"id": m["id"], "state": "queued"},
                    {
                        "id": m["id"],
                        "state": "failed",
                        "error": "bad",
                        "status_code": 422,
                    },
                ]
            }
        )
        socket.push(hello("image", computer="boot-2"))
        socket.push({"state": "models", "models": models(formatting=True)})
        socket.push({"state": "ready", "protocol": 4, "pid": 1})
        with self.assertLogs("api.services.worker_hub", level="ERROR"):
            await self.hub.serve(socket)
        self.assertEqual(socket.closed, 1011)
        self.assertEqual([w.kind for w in self.hub.workers.values()], ["speech"])


class JobTests(HubTestCase):
    async def test_job_runs_on_a_worker_of_its_type_and_resolves(self):
        await self.connect("image")
        socket, worker = await self.connect("lfm3_llm")
        job = await self.hub.submit(chat())
        self.assertEqual(job.machines, {worker.id})
        self.assertEqual(worker.slot.pending, 1)
        memory = {"capacity": [1, 2], "used": [0, 1], "residents": 1}
        socket.push({"id": job.request.id, "state": "running", "worker": "lfm3_llm"})
        socket.push(
            {
                "id": job.request.id,
                "state": "succeeded",
                "worker": "lfm3_llm",
                "result": {"text": "hello"},
                "memory": memory,
            }
        )
        self.assertEqual(await job.result, {"text": "hello"})
        self.assertEqual(worker.slot.pending, 0)
        self.assertFalse(worker.slot.busy)
        self.assertIsNone(
            self.hub.memory
        )  # Per-request LFM does not report native residency memory.
        self.assertEqual(
            [event.state for _, event in self.events],
            ["queued", "running", "succeeded"],
        )

    async def test_a_scaled_worker_type_prefers_the_loaded_copy(self):
        await self.connect("image", instance="image-1")
        _, warm = await self.connect("image", instance="image-2")
        warm.slot.loaded = True
        job = await self.hub.submit(image())
        self.assertEqual(job.machines, {warm.id})

    async def test_rejection_before_acceptance_fails_submission(self):
        await self.connect(
            "lfm3_llm",
            replies={
                "submit": lambda m: [
                    {"id": m["id"], "state": "rejected", "error": "queue_full"}
                ]
            },
        )
        with self.assertRaises(InferenceFailure) as raised:
            await self.hub.submit(chat())
        self.assertEqual(raised.exception.status_code, 429)
        self.assertFalse(self.hub.jobs)

    async def test_streamed_content_is_bounded(self):
        socket, _ = await self.connect("lfm3_llm")
        job = await self.hub.submit(chat(), streaming=True)
        socket.push({"id": job.request.id, "state": "content", "content": "hi"})
        await self.settle()
        self.assertEqual(await job.events.get(), {"content": "hi"})

    async def test_no_worker_of_the_type_is_unavailable(self):
        await self.connect("image")
        with (
            patch("api.services.worker_hub.CONNECT_TIMEOUT", 0.01),
            self.assertRaises(InferenceFailure) as raised,
        ):
            await self.hub.submit(chat())
        self.assertEqual(raised.exception.status_code, 503)

    async def test_a_request_waits_for_its_worker_container_to_connect(self):
        submission = asyncio.create_task(self.hub.submit(chat()))
        await self.settle()
        self.assertFalse(submission.done())
        _, worker = await self.connect("lfm3_llm")
        job = await submission
        self.assertEqual(job.machines, {worker.id})

    async def test_settings_go_to_native_stores_and_settle_once_all_finish(self):
        language, _ = await self.connect("lfm3_llm")
        first, _ = await self.connect("decision")
        second, _ = await self.connect("image")
        job = await self.hub.submit(
            InferenceTask(
                operation="configure",
                input={"feature": "whisper_formatting", "enabled": False},
            )
        )
        self.assertFalse(language.commands("submit"))
        first.push({"id": job.request.id, "state": "succeeded", "result": {}})
        await self.settle()
        self.assertFalse(job.result.done())
        second.push(
            {
                "id": job.request.id,
                "state": "failed",
                "error": "bad",
                "status_code": 422,
            }
        )
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 422)

    async def test_a_download_goes_to_one_worker_of_its_type_per_computer(self):
        store = models(checkpoints=[checkpoint("qwen-image-2.1-turbo", kind="image")])
        language, _ = await self.connect("lfm3_llm", store=store)
        here, _ = await self.connect("image", instance="image-1", store=store)
        twin, _ = await self.connect("image", instance="image-2", store=store)
        there, _ = await self.connect("image", computer="boot-2", store=store)
        await self.hub.submit(
            InferenceTask(
                operation="download",
                input={"feature": "models", "model_id": "qwen-image-2.1-turbo"},
            )
        )
        self.assertFalse(language.commands("submit"))
        self.assertEqual(len(here.commands("submit")) + len(twin.commands("submit")), 1)
        self.assertEqual(len(there.commands("submit")), 1)

    async def test_cancel_reaches_the_worker(self):
        socket, _ = await self.connect("lfm3_llm")
        job = await self.hub.submit(chat())
        await self.hub.cancel(job.request.id)
        self.assertEqual(
            socket.commands("cancel"), [{"id": job.request.id, "command": "cancel"}]
        )


class DisconnectTests(HubTestCase):
    async def test_waiting_job_moves_and_running_job_fails(self):
        first, _ = await self.connect("lfm3_llm", instance="llm-1")
        waiting = await self.hub.submit(chat())
        running = await self.hub.submit(chat())
        first.push({"id": running.request.id, "state": "running"})
        await self.settle()
        second, other = await self.connect("lfm3_llm", instance="llm-2")
        await first.close()
        await self.settle()
        with self.assertRaises(InferenceFailure) as raised:
            await running.result
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            [m["id"] for m in second.commands("submit")], [waiting.request.id]
        )
        self.assertEqual(waiting.machines, {other.id})
        self.assertEqual(other.slot.pending, 1)

    async def test_cancelling_a_job_without_workers_does_not_wait(self):
        await self.connect("lfm3_llm", replies={"cancel": lambda m: []})
        job = await self.hub.submit(chat())
        job.machines = {"gone"}
        await asyncio.wait_for(self.hub.cancel_and_join(job), 1)
        with self.assertRaises(InferenceFailure):
            await job.result

    async def test_pending_controls_fail_on_disconnect(self):
        socket, worker = await self.connect("lfm3_llm")
        control = asyncio.create_task(worker.control("status"))
        await self.settle()
        await socket.close()
        with self.assertRaises(InferenceFailure):
            await control


class CancellationTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.hub = WorkerHub()

    async def asyncTearDown(self):
        for job in self.hub.jobs.values():
            for future in (job.accepted, job.result):
                if future.done() and not future.cancelled():
                    future.exception()

    def add_worker(self, instance, kind="lfm3_llm"):
        info = hello(kind, instance=instance, computer=instance)
        computer = Computer(info["machine"])
        worker = Worker(FakeSocket(), info, computer)
        computer.ledger.workers[worker.id] = WorkerSlot(
            name=worker.id,
            kind=kind,
            devices=worker.devices,
            budget=[0] * len(computer.ledger.capacity),
        )
        worker.ready.set()
        self.hub.computers[computer.id] = computer
        self.hub.workers[worker.id] = worker
        return worker

    async def add_job(self, state=JobState.QUEUED, request=None):
        job = InferenceJob(request or chat(), False)
        job.worker = "lfm3_llm" if request is None else None
        self.hub.jobs[job.request.id] = job
        await self.hub._send_job(job)
        job.accepted.set_result(None)
        job.state = state
        return job

    def event(self, worker, job, state, **fields):
        self.hub._job_event(
            worker, WorkerEvent(id=job.request.id, state=state, **fields)
        )

    def assert_cancelled(self, job):
        self.assertTrue(job.result.done())
        with self.assertRaises(InferenceFailure) as raised:
            job.result.result()
        self.assertEqual(raised.exception.status_code, 499)

    def store_request(self):
        return InferenceTask(
            operation="configure",
            input={"feature": "whisper_formatting", "enabled": False},
        )


class JobSocket(FakeSocket):
    """In-memory delivery: unknown cancellation IDs produce no reply."""

    def __init__(self, hub, worker):
        super().__init__()
        self.hub = hub
        self.worker = worker
        self.active = set()

    async def send_text(self, text):
        await super().send_text(text)
        message = json.loads(text)
        job_id = message["id"]
        if message["command"] == "submit":
            self.active.add(job_id)
            self.hub._job_event(self.worker, WorkerEvent(id=job_id, state="queued"))
        elif message["command"] == "cancel" and job_id in self.active:
            self.finish(job_id, "cancelled")

    def finish(self, job_id, state, **fields):
        assert job_id in self.active
        self.active.remove(job_id)
        self.hub._job_event(self.worker, WorkerEvent(id=job_id, state=state, **fields))


class LockedSubmissionTests(CancellationTestCase):
    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0)

    async def test_caller_cancel_waiting_for_write_lock_retires_undelivered(self):
        worker = self.add_worker("first")
        socket = worker.websocket = JobSocket(self.hub, worker)
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = socket.send_text

        async def send(text):
            if json.loads(text)["command"] == "status":
                entered.set()
                await release.wait()
            await send_text(text)

        with patch.object(socket, "send_text", side_effect=send):
            command = asyncio.create_task(
                worker.send({"id": "control", "command": "status"})
            )
            await entered.wait()
            submission = asyncio.create_task(self.hub.submit(chat()))
            await self.settle()
            job = next(iter(self.hub.jobs.values()))
            submission.cancel()
            await self.settle()
            release.set()
            await command
            try:
                done, _ = await asyncio.wait({submission}, timeout=0.05)
                self.assertIn(
                    submission,
                    done,
                    "undelivered submit waits for an impossible terminal",
                )
                with self.assertRaises(asyncio.CancelledError):
                    await submission
                self.assertEqual(socket.commands("submit"), [])
                self.assertEqual(socket.commands("cancel"), [])
                self.assertEqual(worker.slot.pending, 0)
                self.assertEqual(worker.jobs, set())
                self.assertEqual(job.machines, set())
                self.assertFalse(job.sent)
                self.assertNotIn(job.request.id, self.hub.jobs)
            finally:
                if not submission.done():
                    await self.hub._disconnected(worker)
                await asyncio.gather(submission, return_exceptions=True)
                for future in (job.accepted, job.result):
                    if future.done():
                        future.exception()

    async def test_caller_cancel_inside_write_preserves_atomic_delivery(self):
        worker = self.add_worker("first")
        socket = worker.websocket = JobSocket(self.hub, worker)
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = socket.send_text
        interrupted = []

        async def send(text):
            if json.loads(text)["command"] == "submit":
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.append(True)
                    raise
            await send_text(text)

        with patch.object(socket, "send_text", side_effect=send):
            submission = asyncio.create_task(self.hub.submit(chat()))
            await entered.wait()
            job = next(iter(self.hub.jobs.values()))
            submission.cancel()
            await self.settle()
            submission.cancel()
            await self.settle()
            try:
                self.assertEqual(
                    interrupted, [], "caller cancellation interrupted an open write"
                )
                self.assertTrue(worker.write_lock.locked())
                self.assertTrue(job.cancel_requested)
                self.assertFalse(submission.done() or job.result.done())
                release.set()
                done, _ = await asyncio.wait({submission}, timeout=0.1)
                self.assertIn(submission, done)
                with self.assertRaises(asyncio.CancelledError):
                    await submission
                self.assertEqual(
                    [message["command"] for message in socket.sent],
                    ["submit", "cancel"],
                )
                self.assertEqual(socket.active, set())
                self.assertEqual(worker.slot.pending, 0)
            finally:
                release.set()
                if not submission.done():
                    await self.hub._disconnected(worker)
                await asyncio.gather(submission, return_exceptions=True)
                for future in (job.accepted, job.result):
                    if future.done():
                        future.exception()

    async def test_cancel_intent_while_waiting_for_lock_skips_submit_and_replacement(
        self,
    ):
        worker = self.add_worker("first")
        socket = worker.websocket = JobSocket(self.hub, worker)
        await worker.write_lock.acquire()
        submission = asyncio.create_task(self.hub.submit(chat()))
        try:
            await self.settle()
            job = next(iter(self.hub.jobs.values()))
            await self.hub.cancel(job.request.id)
            self.assertTrue(job.cancel_requested)
            self.assertFalse(job.result.done())
            replacement = self.add_worker("second")
            await self.hub._disconnected(worker)
            await self.hub._send_job(job)
            worker.write_lock.release()
            with self.assertRaises(InferenceFailure) as raised:
                await submission
            self.assertEqual(raised.exception.status_code, 499)
            self.assertEqual(socket.sent, [])
            self.assertEqual(replacement.websocket.sent, [])
            self.assertFalse(job.sent)
            self.assertEqual(job.machines, set())
        finally:
            if worker.write_lock.locked():
                worker.write_lock.release()
            await asyncio.gather(submission, return_exceptions=True)

    async def test_uncertain_write_failure_waits_for_disconnect(self):
        for error in (OSError("write failed"), TimeoutError("write timed out")):
            with self.subTest(error=type(error).__name__):
                worker = self.add_worker(type(error).__name__)
                socket = worker.websocket = JobSocket(self.hub, worker)
                send_text = socket.send_text

                async def send(text, error=error, send_text=send_text):
                    if json.loads(text)["command"] == "submit":
                        raise error
                    await send_text(text)

                with (
                    patch.object(socket, "send_text", side_effect=send),
                    patch.object(self.hub, "_route", return_value=worker),
                ):
                    submission = asyncio.create_task(self.hub.submit(chat()))
                    await self.settle()
                    job = next(iter(self.hub.jobs.values()))
                    try:
                        self.assertTrue(job.cancel_requested)
                        self.assertTrue(job.sent)
                        self.assertEqual(socket.commands("submit"), [])
                        self.assertEqual(len(socket.commands("cancel")), 1)
                        self.assertEqual(socket.active, set())
                        self.assertFalse(submission.done() or job.result.done())
                        self.assertEqual(job.machines, {worker.id})
                        self.assertEqual(worker.jobs, {job.request.id})
                        self.assertEqual(worker.slot.pending, 1)
                        replacement = self.add_worker("replacement")
                        await self.hub._disconnected(worker)
                        with self.assertRaises(type(error)):
                            await submission
                        self.assertEqual(replacement.websocket.sent, [])
                        self.assertEqual(job.machines, set())
                    finally:
                        if not submission.done():
                            await self.hub._disconnected(worker)
                        await asyncio.gather(submission, return_exceptions=True)

    async def test_terminal_failure_waiting_for_lock_inhibits_submission(self):
        worker = self.add_worker("first")
        job = InferenceJob(chat(), False)
        job.worker = "lfm3_llm"
        self.hub.jobs[job.request.id] = job
        await worker.write_lock.acquire()
        submission = asyncio.create_task(self.hub._send_job(job))
        try:
            await self.settle()
            self.hub._fail(job, InferenceFailure("does not fit", 507))
            self.assertFalse(job.result.done())
            await self.hub._send_job(job)
        finally:
            worker.write_lock.release()
            await submission
        self.assertEqual(worker.websocket.sent, [])
        self.assertEqual(worker.slot.pending, 0)
        self.assertFalse(job.sent)
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 507)

    async def test_disconnect_before_write_preserves_disconnection_failure(self):
        worker = self.add_worker("first")
        await worker.write_lock.acquire()
        submission = asyncio.create_task(self.hub.submit(chat()))
        try:
            await self.settle()
            job = next(iter(self.hub.jobs.values()))
            await self.hub._disconnected(worker)
        finally:
            worker.write_lock.release()
        with self.assertRaises(InferenceFailure) as raised:
            await submission
        self.assertEqual(raised.exception.status_code, 503)
        with self.assertRaises(InferenceFailure) as outcome:
            job.result.result()
        self.assertEqual(outcome.exception.status_code, 503)
        self.assertEqual(worker.websocket.sent, [])
        self.assertFalse(job.sent)

    async def test_error_after_delivery_uses_real_cancellation_confirmation(self):
        worker = self.add_worker("first")
        socket = worker.websocket = JobSocket(self.hub, worker)
        send_text = socket.send_text
        jobs = []

        async def send(text):
            await send_text(text)
            message = json.loads(text)
            if message["command"] == "submit":
                jobs.append(self.hub.jobs[message["id"]])
                raise OSError("uncertain completion")

        with (
            patch.object(socket, "send_text", side_effect=send),
            self.assertRaisesRegex(OSError, "uncertain completion"),
        ):
            await self.hub.submit(chat())
        self.assertEqual(
            [message["command"] for message in socket.sent], ["submit", "cancel"]
        )
        self.assertEqual(socket.active, set())
        self.assertEqual(jobs[0].machines, set())
        self.assertEqual(worker.slot.pending, 0)
        self.assertNotIn(jobs[0].request.id, self.hub.jobs)


class CancellationHandoffTests(CancellationTestCase):
    async def test_cancelled_waiting_job_is_not_submitted_to_replacement(self):
        for state in (JobState.QUEUED, JobState.PREPARING, JobState.WAITING):
            with self.subTest(state=state):
                self.hub.workers.clear()
                self.hub.computers.clear()
                first = self.add_worker("first")
                job = await self.add_job(state)
                second = self.add_worker("second")
                await self.hub.cancel(job.request.id)
                await self.hub._disconnected(first)
                self.assertEqual(second.websocket.commands("submit"), [])
                self.assert_cancelled(job)

    async def test_cancel_and_join_waits_for_connected_worker_cleanup(self):
        worker = self.add_worker("first")
        job = await self.add_job(JobState.RUNNING)
        worker.slot.busy = True
        cancellation = asyncio.create_task(self.hub.cancel_and_join(job))
        await asyncio.sleep(0)
        self.assertTrue(job.cancel_requested)
        self.assertFalse(cancellation.done() or job.result.done())
        self.assertTrue(worker.slot.busy)
        self.assertEqual(worker.slot.pending, 1)
        self.event(worker, job, "cancelled")
        await cancellation
        self.assert_cancelled(job)
        self.assertFalse(worker.slot.busy)
        self.assertEqual(worker.slot.pending, 0)

    async def test_disconnect_during_cancel_send_does_not_restart_job(self):
        worker = self.add_worker("first")
        job = await self.add_job()
        replacement = self.add_worker("second")
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(message):
            entered.set()
            await release.wait()

        with patch.object(worker, "send", side_effect=send):
            cancellation = asyncio.create_task(self.hub.cancel_and_join(job))
            await entered.wait()
            await self.hub._disconnected(worker)
            release.set()
            await cancellation
        self.assertEqual(replacement.websocket.commands("submit"), [])
        self.assert_cancelled(job)

    async def test_cancel_during_handoff_send_follows_submission(self):
        first = self.add_worker("first")
        job = await self.add_job()
        second = self.add_worker("second")
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = second.websocket.send_text

        async def send(text):
            if json.loads(text)["command"] == "submit":
                entered.set()
                await release.wait()
            await send_text(text)

        with patch.object(second.websocket, "send_text", side_effect=send):
            handoff = asyncio.create_task(self.hub._disconnected(first))
            await entered.wait()
            cancellation = asyncio.create_task(self.hub.cancel_and_join(job))
            await asyncio.sleep(0)
            self.assertTrue(job.cancel_requested)
            self.assertFalse(job.result.done())
            release.set()
            await handoff
            await self.hub.cancel(job.request.id)
            self.assertEqual(
                [message["command"] for message in second.websocket.sent][:2],
                ["submit", "cancel"],
            )
            self.event(second, job, "cancelled")
            await cancellation
        self.assert_cancelled(job)

    async def test_cancelled_job_is_not_sent_even_without_available_workers(self):
        job = InferenceJob(chat(), False)
        job.worker = "lfm3_llm"
        self.hub.jobs[job.request.id] = job
        await self.hub.cancel_and_join(job)
        await self.hub._send_job(job)
        replacement = self.add_worker("replacement")
        await self.hub._send_job(job)
        self.assertEqual(replacement.websocket.commands("submit"), [])
        self.assertEqual(replacement.slot.pending, 0)
        self.assert_cancelled(job)

    async def test_cancel_without_workers_settles_locally(self):
        worker = self.add_worker("first")
        job = await self.add_job()
        self.hub.workers.pop(worker.id)
        await self.hub.cancel(job.request.id)
        self.assert_cancelled(job)
        self.assertTrue(job.cancel_requested)

    async def test_repeated_cancellation_keeps_ownership_until_terminal_event(self):
        worker = self.add_worker("first")
        job = await self.add_job()
        await self.hub.cancel(job.request.id)
        await self.hub.cancel(job.request.id)
        await self.hub._send_job(job)
        self.assertFalse(job.result.done())
        self.assertEqual(len(worker.websocket.commands("submit")), 1)
        self.assertEqual(job.machines, {worker.id})
        self.assertEqual(worker.slot.pending, 1)
        self.event(worker, job, "cancelled")
        await self.hub.cancel(job.request.id)
        await self.hub.cancel_and_join(job)
        self.assertEqual(len(worker.websocket.commands("cancel")), 2)
        self.assert_cancelled(job)
        self.assertEqual(worker.slot.pending, 0)

    async def test_uncancelled_handoff_rejects_old_worker_events(self):
        for state in (JobState.QUEUED, JobState.PREPARING, JobState.WAITING):
            with self.subTest(state=state):
                self.hub.workers.clear()
                self.hub.computers.clear()
                first = self.add_worker("first")
                job = await self.add_job(state)
                second = self.add_worker("second")
                await self.hub._disconnected(first)
                self.assertEqual(len(second.websocket.commands("submit")), 1)
                self.assertEqual(job.machines, {second.id})
                self.event(first, job, "cancelled")
                self.event(first, job, "succeeded", result={"text": "stale"})
                self.assertFalse(job.result.done())
                self.assertEqual(second.slot.pending, 1)
                self.event(second, job, "succeeded", result={"text": "current"})
                self.assertEqual(await job.result, {"text": "current"})

    async def test_cancelled_store_waits_for_all_remaining_workers(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        job = await self.add_job(request=self.store_request())
        await self.hub.cancel(job.request.id)
        self.assertEqual(len(first.websocket.commands("cancel")), 1)
        self.assertEqual(len(second.websocket.commands("cancel")), 1)
        await self.hub._disconnected(first)
        self.assertFalse(job.result.done())
        self.assertEqual(job.machines, {second.id})
        self.event(first, job, "succeeded", result={"stale": True})
        self.event(second, job, "cancelled")
        self.assert_cancelled(job)
        self.assertEqual(len(second.websocket.commands("submit")), 1)

    async def test_cancel_during_store_submission_skips_unsent_workers(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = first.websocket.send_text

        async def send(text):
            if json.loads(text)["command"] == "submit":
                entered.set()
                await release.wait()
            await send_text(text)

        with patch.object(first.websocket, "send_text", side_effect=send):
            submission = asyncio.create_task(self.add_job(request=self.store_request()))
            await entered.wait()
            job = next(iter(self.hub.jobs.values()))
            cancellation = asyncio.create_task(self.hub.cancel(job.request.id))
            await asyncio.sleep(0)
            self.assertTrue(job.cancel_requested)
            release.set()
            await submission
            await cancellation
        self.assertEqual(second.websocket.commands("submit"), [])
        self.assertEqual(job.machines, {first.id})
        self.assertFalse(job.result.done())
        self.event(first, job, "cancelled")
        self.assert_cancelled(job)

    async def test_failed_cancel_send_keeps_connected_worker_ownership(self):
        worker = self.add_worker("first")
        job = await self.add_job(JobState.RUNNING)
        worker.slot.busy = True
        with patch.object(worker, "send", side_effect=OSError("send failed")) as send:
            cancellation = asyncio.create_task(self.hub.cancel_and_join(job))
            await asyncio.sleep(0)
            send.assert_awaited_once()
            self.assertFalse(cancellation.done() or job.result.done())
            self.assertTrue(worker.slot.busy)
            self.assertEqual(worker.slot.pending, 1)
            self.event(worker, job, "cancelled")
            await cancellation
        self.assert_cancelled(job)

    async def test_store_cancel_attempts_other_workers_after_a_send_failure(self):
        workers = [self.add_worker("first", "decision"), self.add_worker("second", "image")]
        job = await self.add_job(request=self.store_request())
        first, second = [self.hub.workers[worker_id] for worker_id in job.machines]
        with (
            patch.object(first, "send", side_effect=OSError("send failed")),
            self.assertRaises(OSError),
        ):
            await self.hub.cancel(job.request.id)
        self.assertEqual(len(second.websocket.commands("cancel")), 1)
        self.assertFalse(job.result.done())
        for worker in workers:
            self.event(worker, job, "cancelled")
        self.assert_cancelled(job)

    async def test_send_disconnect_cancels_remaining_store_workers_and_joins(self):
        self.add_worker("first", "decision")
        self.add_worker("second", "image")
        job = await self.add_job(request=self.store_request())
        first, second = [self.hub.workers[worker_id] for worker_id in job.machines]
        with patch.object(first, "send", side_effect=WebSocketDisconnect(1006)):
            cancellation = asyncio.create_task(self.hub.cancel_and_join(job))
            try:
                for _ in range(20):
                    await asyncio.sleep(0)
                self.assertEqual(len(second.websocket.commands("cancel")), 1)
                self.assertFalse(cancellation.done() or job.result.done())
                self.assertEqual(job.machines, {first.id, second.id})
                await self.hub._disconnected(first)
                self.assertFalse(cancellation.done() or job.result.done())
                self.assertEqual(job.machines, {second.id})
                self.event(second, job, "cancelled")
                await cancellation
                self.assert_cancelled(job)
            finally:
                for worker_id in tuple(job.machines):
                    self.event(self.hub.workers[worker_id], job, "cancelled")
                await asyncio.gather(cancellation, return_exceptions=True)


class SubmissionOwnershipTests(CancellationTestCase):
    def register_job(self, request=None):
        job = InferenceJob(request or self.store_request(), False)
        job.worker = "lfm3_llm" if request is not None else None
        self.hub.jobs[job.request.id] = job
        return job

    async def test_early_terminal_does_not_complete_remaining_fanout(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        job = self.register_job()

        async def send(message):
            self.event(first, job, "succeeded", result={"first": True})

        with patch.object(first.websocket, "send_text", side_effect=send):
            await self.hub._send_job(job)
        self.assertFalse(job.result.done())
        self.assertEqual(job.machines, {second.id})
        self.event(second, job, "failed", error="bad settings", status_code=422)
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 422)

    async def test_all_early_terminals_wait_for_submission_to_close(self):
        workers = [self.add_worker("first", "decision"), self.add_worker("second", "image")]
        job = self.register_job()
        completed_during_send = []

        async def first_send(message):
            self.event(workers[0], job, "succeeded", result={"worker": workers[0].name})
            completed_during_send.append(job.result.done())

        async def second_send(message):
            self.event(workers[1], job, "succeeded", result={"worker": workers[1].name})
            completed_during_send.append(job.result.done())

        with (
            patch.object(workers[0].websocket, "send_text", side_effect=first_send),
            patch.object(workers[1].websocket, "send_text", side_effect=second_send),
        ):
            await self.hub._send_job(job)
        self.assertEqual(completed_during_send, [False, False])
        self.assertEqual(await job.result, {"worker": workers[0].name})
        self.assertEqual(job.machines, set())

    async def test_cancel_after_early_terminal_stops_unsent_targets(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        third = self.add_worker("third", "video")
        job = self.register_job()
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = second.websocket.send_text

        async def first_send(message):
            self.event(first, job, "succeeded", result={})

        async def second_send(text):
            if json.loads(text)["command"] == "submit":
                entered.set()
                await release.wait()
            await send_text(text)

        with (
            patch.object(first.websocket, "send_text", side_effect=first_send),
            patch.object(second.websocket, "send_text", side_effect=second_send),
        ):
            submission = asyncio.create_task(self.hub._send_job(job))
            await entered.wait()
            cancellation = asyncio.create_task(self.hub.cancel(job.request.id))
            await asyncio.sleep(0)
            intent_recorded = job.cancel_requested
            completed_before_release = job.result.done()
            ownership = set(job.machines)
            release.set()
            await submission
            await self.hub.cancel(job.request.id)
            completed_before_terminal = job.result.done()
            self.event(second, job, "cancelled")
            await asyncio.wait_for(cancellation, 1)
            await self.hub.cancel_and_join(job)
        self.assertTrue(intent_recorded)
        self.assertFalse(completed_before_release or completed_before_terminal)
        self.assertEqual(ownership, {second.id})
        self.assertEqual(third.websocket.sent, [])
        self.assertEqual(first.websocket.commands("cancel"), [])
        self.assertEqual(
            [message["command"] for message in second.websocket.sent],
            ["submit", "cancel", "cancel"],
        )
        self.assert_cancelled(job)

    async def test_cancelled_partial_fanout_cannot_succeed(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        third = self.add_worker("third", "video")
        job = self.register_job()
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = second.websocket.send_text

        async def first_send(message):
            self.event(first, job, "succeeded", result={"first": True})

        async def second_send(text):
            if json.loads(text)["command"] == "submit":
                entered.set()
                await release.wait()
            await send_text(text)

        with (
            patch.object(first.websocket, "send_text", side_effect=first_send),
            patch.object(second.websocket, "send_text", side_effect=second_send),
        ):
            submission = asyncio.create_task(self.hub._send_job(job))
            await entered.wait()
            cancellation = asyncio.create_task(self.hub.cancel(job.request.id))
            await asyncio.sleep(0)
            self.assertTrue(job.cancel_requested)
            release.set()
            await submission
            await cancellation
        self.assertEqual(third.websocket.sent, [])
        self.assertFalse(job.result.done())
        self.event(second, job, "succeeded", result={"second": True})
        self.assert_cancelled(job)

    async def test_deferred_terminal_failure_inhibits_requeue(self):
        first = self.add_worker("first")
        second = self.add_worker("second")
        job = self.register_job(chat())

        async def send(message):
            self.hub._fail(job, InferenceFailure("does not fit", 507))
            await self.hub._disconnected(first)

        with (
            patch.object(first.websocket, "send_text", side_effect=send),
            patch.object(self.hub, "_route", side_effect=[first, second]),
        ):
            await self.hub._send_job(job)
        self.assertEqual(second.websocket.commands("submit"), [])
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 507)

    async def test_cancel_preserves_prior_store_failure_while_other_send_is_open(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        third = self.add_worker("third", "video")
        job = self.register_job()
        entered, release = asyncio.Event(), asyncio.Event()
        send_text = second.websocket.send_text

        async def send(text):
            if json.loads(text)["command"] == "submit":
                entered.set()
                await release.wait()
            await send_text(text)

        with patch.object(second.websocket, "send_text", side_effect=send):
            submission = asyncio.create_task(self.hub._send_job(job))
            await entered.wait()
            self.event(first, job, "failed", error="bad settings", status_code=422)
            cancellation = asyncio.create_task(self.hub.cancel(job.request.id))
            await asyncio.sleep(0)
            release.set()
            await submission
            await cancellation
        self.assertFalse(job.result.done())
        self.event(second, job, "succeeded", result={})
        self.assertEqual(third.websocket.sent, [])
        with self.assertRaisesRegex(InferenceFailure, "bad settings") as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 422)

    async def test_disconnected_unsent_store_target_is_incomplete(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        job = self.register_job()

        async def send(text):
            self.event(first, job, "succeeded", result={})
            await self.hub._disconnected(second)

        with patch.object(first.websocket, "send_text", side_effect=send):
            await self.hub._send_job(job)
        self.assertEqual(second.websocket.sent, [])
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 503)

    async def test_cancel_with_no_recipients_waits_for_open_send(self):
        worker = self.add_worker("first", "decision")
        job = self.register_job()
        completed_during_send = []

        async def send(message):
            self.event(worker, job, "succeeded", result={})
            await self.hub.cancel(job.request.id)
            completed_during_send.append(job.result.done())

        with patch.object(worker.websocket, "send_text", side_effect=send):
            await self.hub._send_job(job)
        self.assertEqual(completed_during_send, [False])
        self.assertTrue(job.cancel_requested)
        self.assert_cancelled(job)

    async def test_cancelled_disconnect_during_send_waits_for_send_to_close(self):
        worker = self.add_worker("first")
        job = self.register_job(chat())
        completed_during_send = []
        cancellations = []

        async def send(text):
            cancellations.append(asyncio.create_task(self.hub.cancel(job.request.id)))
            await asyncio.sleep(0)
            await self.hub._disconnected(worker)
            completed_during_send.append(job.result.done())

        with patch.object(worker.websocket, "send_text", side_effect=send):
            await self.hub._send_job(job)
        await asyncio.gather(*cancellations)
        self.assertEqual(completed_during_send, [False])
        self.assert_cancelled(job)
        self.assertEqual(job.machines, set())

    async def test_overlapping_requeue_waits_for_original_send_to_close(self):
        first = self.add_worker("first")
        second = self.add_worker("second")
        job = self.register_job(chat())
        completed_during_send = []

        async def first_send(message):
            await self.hub._disconnected(first)
            completed_during_send.append(job.result.done())

        async def second_send(message):
            self.event(second, job, "succeeded", result={"replacement": True})
            completed_during_send.append(job.result.done())

        with (
            patch.object(first.websocket, "send_text", side_effect=first_send),
            patch.object(second.websocket, "send_text", side_effect=second_send),
            patch.object(self.hub, "_route", side_effect=[first, second]),
        ):
            await self.hub._send_job(job)
        self.assertEqual(completed_during_send, [False, False])
        self.assertEqual(await job.result, {"replacement": True})
        self.assertEqual(second.slot.pending, 0)
        self.assertEqual(job.machines, set())

    async def test_cancel_during_overlapping_requeue_waits_for_both_sends(self):
        first = self.add_worker("first")
        second = self.add_worker("second")
        job = self.register_job(chat())
        completed_during_send = []
        cancellations = []

        async def first_send(message):
            await self.hub._disconnected(first)
            completed_during_send.append(job.result.done())

        async def second_send(text):
            if json.loads(text)["command"] == "cancel":
                return
            cancellations.append(asyncio.create_task(self.hub.cancel(job.request.id)))
            await asyncio.sleep(0)
            self.event(second, job, "cancelled")
            completed_during_send.append(job.result.done())

        with (
            patch.object(first.websocket, "send_text", side_effect=first_send),
            patch.object(second.websocket, "send_text", side_effect=second_send),
            patch.object(self.hub, "_route", side_effect=[first, second]),
        ):
            await self.hub._send_job(job)
            await asyncio.gather(*cancellations)
        self.assertEqual(completed_during_send, [False, False])
        self.assert_cancelled(job)
        self.assertEqual(second.slot.pending, 0)

    async def test_disconnect_without_replacement_defers_failure_until_send_closes(
        self,
    ):
        worker = self.add_worker("first")
        job = self.register_job(chat())
        completed_during_send = []

        async def send(message):
            await self.hub._disconnected(worker)
            completed_during_send.append(job.result.done())

        with patch.object(worker.websocket, "send_text", side_effect=send):
            await self.hub._send_job(job)
        self.assertEqual(completed_during_send, [False])
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 503)

    async def test_partial_send_failure_keeps_submission_until_all_cleanup(self):
        first = self.add_worker("first", "decision")
        second = self.add_worker("second", "image")
        third = self.add_worker("third", "video")
        first_socket = first.websocket = JobSocket(self.hub, first)
        second_socket = second.websocket = JobSocket(self.hub, second)
        send_text = second_socket.send_text
        finish_first, finish_second = first_socket.finish, second_socket.finish

        async def second_send(text):
            await send_text(text)
            if json.loads(text)["command"] == "submit":
                raise OSError("partial send")

        with (
            patch.object(second_socket, "send_text", side_effect=second_send),
            patch.object(first_socket, "finish") as first_cleanup,
            patch.object(second_socket, "finish") as second_cleanup,
        ):
            submission = asyncio.create_task(self.hub.submit(self.store_request()))
            for _ in range(30):
                await asyncio.sleep(0)
            job = next(iter(self.hub.jobs.values()))
            self.assertEqual(job.machines, {first.id, second.id})
            first_cleanup.assert_called_once_with(job.request.id, "cancelled")
            second_cleanup.assert_called_once_with(job.request.id, "cancelled")
            self.assertFalse(job.result.done() or submission.done())
            finish_first(job.request.id, "cancelled")
            self.assertFalse(job.result.done() or submission.done())
            finish_second(job.request.id, "cancelled")
            with self.assertRaisesRegex(OSError, "partial send"):
                await submission
        for socket in (first_socket, second_socket):
            self.assertEqual(
                [message["command"] for message in socket.sent], ["submit", "cancel"]
            )
            self.assertEqual(socket.active, set())
        self.assertEqual(third.websocket.sent, [])
        self.assertNotIn(job.request.id, self.hub.jobs)

    async def test_no_worker_failure_does_not_fabricate_success(self):
        job = self.register_job(chat())
        with self.assertRaises(InferenceFailure):
            await self.hub._send_job(job)
        self.assertFalse(job.result.done())
        await asyncio.wait_for(self.hub.cancel_and_join(job), 1)
        self.assert_cancelled(job)
        self.assertEqual(job.machines, set())


class ModelIdleSettingsTests(unittest.TestCase):
    def test_default_and_configured_period(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(WorkerSettings(_env_file=None).model_idle_seconds, 3600)
        with patch.dict(os.environ, {"HIKARI_MODEL_IDLE_SECONDS": "120"}):
            self.assertEqual(WorkerSettings(_env_file=None).model_idle_seconds, 120)

    def test_period_must_be_positive_whole_seconds(self):
        for value in ("0", "-1", "1.5", "nan", "inf", "later"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"HIKARI_MODEL_IDLE_SECONDS": value}),
                self.assertRaises(ValidationError),
            ):
                WorkerSettings(_env_file=None)


class KvIdleSettingsTests(unittest.TestCase):
    def test_default_and_configured_period_are_independent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(WorkerSettings(_env_file=None).kv_cache_idle_seconds, 3600)
        with patch.dict(
            os.environ,
            {"HIKARI_KV_CACHE_IDLE_SECONDS": "60", "HIKARI_MODEL_IDLE_SECONDS": "120"},
        ):
            settings = WorkerSettings(_env_file=None)
            self.assertEqual(settings.kv_cache_idle_seconds, 60)
            self.assertEqual(settings.model_idle_seconds, 120)

    def test_period_must_be_positive_whole_seconds(self):
        for value in ("0", "-1", "1.5", "nan", "inf", "later"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"HIKARI_KV_CACHE_IDLE_SECONDS": value}),
                self.assertRaises(ValidationError),
            ):
                WorkerSettings(_env_file=None)


class ModelIdleTests(HubTestCase):
    async def resident(self, kind="lfm3_llm", **identity):
        socket, worker = await self.connect(kind, replies={"offload": ok}, **identity)
        slot = worker.slot
        slot.loaded, slot.budget, slot.last_used = True, [GIB, GIB], 100
        slot.requirement = Requirement("m", (GIB, GIB), (GIB, GIB))
        return socket, worker

    async def test_every_modality_expires_at_the_boundary_and_waits_for_cleanup(self):
        for kind in (
            "lfm3_llm",
            "decision",
            "tts",
            "speech",
            "formatting",
            "image",
            "video",
        ):
            with self.subTest(kind=kind):
                socket, worker = await self.resident(kind, computer=kind)
                with patch("api.services.worker_hub.time.monotonic", return_value=3699):
                    await self.hub._schedule(worker.computer)
                self.assertFalse(socket.commands("offload"))
                with patch("api.services.worker_hub.time.monotonic", return_value=3700):
                    await self.hub._schedule(worker.computer)
                self.assertEqual(len(socket.commands("offload")), 1)
                self.assertTrue(worker.slot.loaded and worker.slot.releasing)
                self.assertEqual(worker.slot.budget, [GIB, GIB])
                socket.push(
                    {
                        "state": "worker",
                        "event": "offloaded",
                        "worker": kind,
                        "loaded": False,
                        "budget": [0, 0],
                    }
                )
                await self.settle()
                await self.hub._schedule(worker.computer)
                self.assertFalse(worker.slot.loaded or worker.slot.releasing)
                self.assertEqual(worker.slot.budget, [0, 0])
                self.assertFalse(socket.commands("load"))

    async def test_idle_clock_restarts_when_each_request_finishes(self):
        socket, worker = await self.resident()
        for completed_at in (5000, 7000):
            job = await self.hub.submit(chat())
            socket.push(
                {"id": job.request.id, "state": "running", "worker": "lfm3_llm"}
            )
            await self.settle()
            with patch(
                "api.services.worker_hub.time.monotonic", return_value=completed_at
            ):
                await self.hub._schedule(worker.computer)
                self.assertFalse(socket.commands("offload"))
                socket.push(
                    {
                        "id": job.request.id,
                        "state": "succeeded",
                        "worker": "lfm3_llm",
                        "result": {},
                    }
                )
                await job.result
            self.assertEqual(worker.slot.last_used, completed_at)
        with patch("api.services.worker_hub.time.monotonic", return_value=10599):
            await self.hub._schedule(worker.computer)
        self.assertFalse(socket.commands("offload"))
        with patch("api.services.worker_hub.time.monotonic", return_value=10600):
            await self.hub._schedule(worker.computer)
        self.assertEqual(len(socket.commands("offload")), 1)

    async def test_busy_pending_unloaded_and_releasing_workers_do_not_expire(self):
        for field, value in (
            ("busy", True),
            ("pending", 1),
            ("loaded", False),
            ("releasing", True),
        ):
            with self.subTest(field=field):
                socket, worker = await self.resident(computer=field)
                setattr(worker.slot, field, value)
                worker.slot.releasing_since = 3700
                with patch("api.services.worker_hub.time.monotonic", return_value=3700):
                    await self.hub._schedule(worker.computer)
                self.assertFalse(socket.commands("offload"))

    async def test_another_workers_queued_request_takes_precedence(self):
        socket, worker = await self.resident()
        _, picture = await self.resident("image")
        await self.hub.submit(image())
        with patch("api.services.worker_hub.time.monotonic", return_value=3700):
            await self.hub._schedule(worker.computer)
        self.assertEqual(picture.slot.pending, 1)
        self.assertFalse(socket.commands("offload"))

    async def test_request_arriving_during_requirement_query_prevents_expiry(self):
        socket, worker = await self.resident()
        worker.slot.requirement = None

        async def requirement(current):
            await self.hub.submit(chat())
            return Requirement("m", (GIB, GIB), (GIB, GIB))

        with (
            patch.object(self.hub, "_requirement", side_effect=requirement),
            patch("api.services.worker_hub.time.monotonic", return_value=3700),
        ):
            await self.hub._schedule(worker.computer)
        self.assertEqual(worker.slot.pending, 1)
        self.assertFalse(socket.commands("offload"))


class SchedulingTests(HubTestCase):
    async def test_waiting_worker_gets_a_budget(self):
        _, worker = await self.connect(
            "lfm3_llm", replies={"requirement": needs((1, 4), (2, 30)), "load": ok}
        )
        await self.hub.submit(chat())
        await self.hub._schedule(worker.computer)
        self.assertTrue(worker.slot.loaded)
        self.assertEqual(worker.slot.budget, [2 * GIB, 24 * GIB])

    async def test_image_load_leaves_room_for_the_language_model(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        await self.connect("lfm3_llm", replies={"requirement": needs((1, 6), (2, 30))})
        await self.connect("decision", replies={"requirement": needs((1, 0), (1, 0))})
        _, picture = await self.connect(
            "image", replies={"requirement": needs((1, 2), (1, 40)), "load": load}
        )
        await self.hub.submit(image())
        await self.hub._schedule(picture.computer)
        self.assertEqual(loads, [[GIB, 18 * GIB]])

    async def test_busy_reply_leaves_the_ledger_unchanged(self):
        image_socket, picture = await self.connect(
            "image", replies={"offload": busy, "load": busy}
        )
        picture.slot.loaded, picture.slot.budget = True, [GIB, 24 * GIB]
        picture.slot.requirement = Requirement("i", (GIB, 2 * GIB), (GIB, 24 * GIB))
        _, language = await self.connect("lfm3_llm")
        language.slot.pending = 1
        language.slot.requirement = Requirement("m", (GIB, 10 * GIB), (GIB, 10 * GIB))
        await self.hub._schedule(language.computer)
        self.assertTrue(picture.slot.loaded)
        self.assertFalse(language.slot.loaded)
        self.assertEqual(
            len(image_socket.commands("offload") + image_socket.commands("load")), 1
        )

    async def test_failed_requirement_still_lets_preparation_report(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        _, worker = await self.connect(
            "lfm3_llm",
            replies={
                "requirement": lambda m: [
                    {
                        "id": m["id"],
                        "state": "rejected",
                        "error": "no tier",
                        "status_code": 409,
                    }
                ],
                "load": load,
            },
        )
        await self.hub.submit(chat())
        with self.assertLogs("api.services.worker_hub", level="INFO"):
            await self.hub._schedule(worker.computer)
        self.assertEqual(loads, [[0, 0]])

    async def test_job_that_can_never_fit_fails_instead_of_waiting(self):
        socket, worker = await self.connect(
            "lfm3_llm", replies={"requirement": needs((1, 200), (1, 200), model="huge")}
        )
        job = await self.hub.submit(chat())
        with self.assertLogs("api.services.worker_hub", level="WARNING"):
            await self.hub._schedule(worker.computer)
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 507)
        self.assertIn("huge needs at least", str(raised.exception))
        self.assertEqual(socket.commands("cancel")[0]["id"], job.request.id)
        self.assertEqual(worker.slot.pending, 0)
        self.assertFalse(socket.commands("load"))

    async def test_worker_events_update_the_ledger_in_its_order(self):
        await self.connect("lfm3_llm", gpus=(("GPU-a", 24), ("GPU-b", 24)))
        socket, picture = await self.connect("image", gpus=(("GPU-b", 24),))
        picture.slot.loaded = True
        socket.push(
            {
                "state": "worker",
                "event": "resized",
                "worker": "image",
                "loaded": True,
                "budget": [GIB, 3 * GIB],
            }
        )
        await self.settle()
        self.assertEqual(picture.slot.budget, [GIB, 0, 3 * GIB])

    async def test_unload_offloads_every_worker_of_the_type(self):
        socket, worker = await self.connect("lfm3_llm", replies={"offload": ok})
        slot = worker.slot
        slot.loaded, slot.budget = True, [GIB, GIB]
        await self.hub.offload("lfm3_llm")
        # The charge stays until the worker confirms the memory is free.
        self.assertTrue(slot.loaded and slot.releasing)
        self.assertEqual(slot.budget, [GIB, GIB])
        socket.push(
            {
                "state": "worker",
                "event": "offloaded",
                "worker": "lfm3_llm",
                "loaded": False,
                "budget": [0, 0],
            }
        )
        await self.settle()
        self.assertFalse(slot.loaded or slot.releasing)
        self.assertEqual(slot.budget, [0, 0])


class ResidentRequirementTests(HubTestCase):
    async def test_routine_snapshots_keep_requirements_and_changes_drop_them(self):
        socket, worker = await self.connect("lfm3_llm", store=models(formatting=True))
        worker.slot.requirement = Requirement("m", (GIB, GIB), (GIB, GIB))
        socket.push({"state": "models", "models": models(formatting=True)})
        await self.settle()
        self.assertIsNotNone(worker.slot.requirement)
        socket.push({"state": "models", "models": models(formatting=False)})
        await self.settle()
        self.assertIsNone(worker.slot.requirement)

    async def test_stale_grant_is_replaced_once_the_requirement_grows(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        _, worker = await self.connect(
            "lfm3_llm", replies={"requirement": needs((1, 6), (2, 30)), "load": load}
        )
        # A grant made before a tier existed: loaded with nothing.
        worker.slot.loaded, worker.slot.budget, worker.slot.pending = True, [0, 0], 1
        await self.hub._schedule(worker.computer)
        self.assertEqual(loads, [[2 * GIB, 24 * GIB]])

    async def test_shrink_waits_for_the_release_before_loading_another_worker(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        llm_socket, language = await self.connect("lfm3_llm", replies={"load": load})
        language.slot.loaded, language.slot.budget = True, [2 * GIB, 24 * GIB]
        language.slot.requirement = Requirement(
            "m", (GIB, 6 * GIB), (2 * GIB, 30 * GIB)
        )
        _, picture = await self.connect("image", replies={"load": load})
        picture.slot.pending = 1
        picture.slot.requirement = Requirement("i", (GIB, 2 * GIB), (GIB, 40 * GIB))
        computer = language.computer
        await self.hub._schedule(computer)
        # Only VRAM is short, so the language model keeps its RAM.
        self.assertEqual(loads, [[2 * GIB, 6 * GIB]])
        self.assertTrue(language.slot.releasing)
        self.assertEqual(language.slot.budget, [2 * GIB, 24 * GIB])
        # Cleanup is slow: nothing else is planned on the unreleased memory.
        await self.hub._schedule(computer)
        self.assertEqual(len(loads), 1)
        llm_socket.push(
            {
                "state": "worker",
                "event": "resized",
                "worker": "lfm3_llm",
                "loaded": True,
                "budget": [2 * GIB, 6 * GIB],
            }
        )
        await self.settle()
        await self.hub._schedule(computer)
        self.assertEqual(loads[1:], [[GIB, 18 * GIB]])


class ReleaseConfirmationTests(HubTestCase):
    async def test_confirmation_before_the_reply_is_not_lost(self):
        def offload(message):
            # The worker finishes cleanup before its reply reaches the API.
            return [
                {
                    "state": "worker",
                    "event": "offloaded",
                    "worker": "image",
                    "loaded": False,
                    "budget": [0, 0],
                },
                *ok(message),
            ]

        _, worker = await self.connect("image", replies={"offload": offload})
        worker.slot.loaded, worker.slot.budget = True, [GIB, 10 * GIB]
        await self.hub.offload("image")
        await self.settle()
        self.assertFalse(worker.slot.releasing or worker.slot.loaded)
        self.assertEqual(worker.slot.budget, [0, 0])

    async def resync(self, status):
        def reply(message):
            return [
                {
                    "id": message["id"],
                    "state": "status",
                    "machine": {},
                    "workers": [{"worker": "store"}, {"worker": "lfm3_llm", **status}],
                }
            ]

        _, worker = await self.connect("lfm3_llm", replies={"status": reply})
        slot = worker.slot
        slot.loaded, slot.budget, slot.releasing = True, [4 * GIB, 20 * GIB], True
        slot.releasing_since = expired()
        with self.assertLogs("api.services.worker_hub", level="WARNING"):
            await self.hub._schedule(worker.computer)
        return slot

    async def test_unconfirmed_release_is_reread_from_status(self):
        slot = await self.resync({"loaded": False, "budget": [0, 0]})
        self.assertFalse(slot.loaded or slot.releasing)
        self.assertEqual(slot.budget, [0, 0])

    async def test_timeout_keeps_the_charge_while_cleanup_continues(self):
        slot = await self.resync(
            {
                "loaded": True,
                "releasing": True,
                "budget": [GIB, 6 * GIB],
                "used": [4 * GIB, 20 * GIB],
            }
        )
        self.assertTrue(slot.releasing)
        self.assertEqual(slot.budget, [4 * GIB, 20 * GIB])

    async def test_timeout_charges_what_the_engine_still_holds(self):
        slot = await self.resync(
            {
                "loaded": True,
                "releasing": False,
                "budget": [GIB, 6 * GIB],
                "used": [GIB, 9 * GIB],
            }
        )
        self.assertFalse(slot.releasing)
        self.assertEqual(slot.budget, [GIB, 9 * GIB])


class ContextConfigurationTests(HubTestCase):
    async def test_manual_preload_is_rejected_without_submitting_or_granting(self):
        socket, _ = await self.connect("lfm3_llm", replies={"load": ok})
        with self.assertRaisesRegex(InferenceFailure, "per request"):
            await self.hub.submit(
                InferenceTask(
                    operation="load", input={"feature": "llm", "context_limit": 8192}
                )
            )
        self.assertFalse(socket.commands("submit"))
        self.assertFalse(socket.commands("load"))
        self.assertFalse(self.hub.jobs)

    async def test_saved_context_is_configured_only_on_the_owning_model_worker(self):
        picture, _ = await self.connect("image")
        language, worker = await self.connect(
            "lfm3_llm", replies={"submit": configured}
        )
        request = InferenceTask(
            operation="configure",
            input={
                "feature": "model_context",
                "model_id": "lfm3-1.7b-bf16",
                "context_limit": 8192,
            },
        )
        job = await self.hub.submit(request)
        await job.result
        self.assertFalse(picture.commands("submit"))
        self.assertEqual(
            language.commands("submit")[0]["input"],
            {
                "feature": "model_context",
                "model_id": "lfm3-1.7b-bf16",
                "context_limit": 8192,
            },
        )
        self.assertEqual(worker.slot.pending, 0)
        self.assertFalse(language.commands("load"))

    async def test_maximum_context_choice_is_not_replaced_with_a_numeric_default(self):
        socket, _ = await self.connect("lfm3_llm", replies={"submit": configured})
        job = await self.hub.submit(
            InferenceTask(
                operation="configure",
                input={
                    "feature": "model_context",
                    "model_id": "lfm3-1.7b-bf16",
                    "context_limit": None,
                },
            )
        )
        await job.result
        self.assertIn("context_limit", socket.commands("submit")[0]["input"])
        self.assertIsNone(socket.commands("submit")[0]["input"]["context_limit"])


class SingleAudioCallScopeTests(unittest.TestCase):
    def test_overlapping_call_scopes_track_one_audio_checkpoint_and_cleanup_on_error(
        self,
    ):
        hub = WorkerHub()
        slot = WorkerSlot(
            name="audio",
            kind="lfm3_audio",
            budget=[0, 0],
            requirement=Requirement(
                model="lfm3-audio-2b-bf16", minimum=(1, 1), preferred=(1, 1)
            ),
        )
        hub.computers["box"] = SimpleNamespace(
            ledger=MachineLedger([100, 100], {"audio": slot})
        )
        with self.assertRaisesRegex(RuntimeError, "disconnect"):
            with hub.full_duplex(frozenset(("lfm3-audio-2b-bf16",))):
                self.assertEqual(len(hub.calls), 1)
                self.assertEqual(slot.priority, 1)
                self.assertEqual(slot.budget, [0, 0])
                self.assertFalse(slot.loaded)
                with hub.full_duplex(frozenset(("lfm3-audio-2b-bf16",))):
                    self.assertEqual(len(hub.calls), 2)
                self.assertEqual(len(hub.calls), 1)
                raise RuntimeError("disconnect")
        self.assertFalse(hub.calls)
        self.assertEqual(slot.priority, 1)
        self.assertEqual(slot.budget, [0, 0])

    def test_audio_call_scope_cannot_pin_an_unrelated_media_checkpoint(self):
        hub = WorkerHub()
        slot = WorkerSlot(
            name="image",
            kind="image",
            budget=[0, 0],
            requirement=Requirement(
                model="qwen-image-2.1-turbo", minimum=(1, 1), preferred=(1, 1)
            ),
        )
        hub.computers["box"] = SimpleNamespace(
            ledger=MachineLedger([100, 100], {"image": slot})
        )
        with hub.full_duplex(frozenset(("lfm3-audio-2b-bf16",))):
            self.assertFalse(slot.full_duplex)
            self.assertEqual(slot.priority, 4)
            self.assertEqual(slot.budget, [0, 0])
