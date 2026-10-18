import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect

from api.pydantic_models.inference_task import InferenceTask
from api.services.inference_errors import InferenceFailure
from api.services.memory_scheduler import GIB, Requirement
from api.services.worker_hub import WorkerHub, authorized, requested_model

WORKERS = ["store", "llm", "decision", "tts", "speech", "formatting", "image", "video"]


class FakeSocket:
    """A worker machine's connection, seen from the API."""

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


def hello(name, workers=WORKERS, protocol=3):
    return {
        "state": "hello",
        "protocol": protocol,
        "pid": 1,
        "machine": {
            "name": name,
            "ram_total": 68 * GIB,
            "ram_free": 64 * GIB,
            "devices": [{"index": 0, "vram_total": 25 * GIB, "vram_free": 24 * GIB}],
        },
        "workers": workers,
    }


def models(tier=None, whisper=None):
    return {
        "models": [],
        "selected_tier": tier,
        "selected_whisper": whisper,
        "whisper_formatting": True,
    }


def queued(message):
    return [{"id": message["id"], "state": "queued"}]


def chat():
    return InferenceTask(
        operation="completion",
        input={"feature": "llm", "messages": [{"role": "user", "text": "hi"}]},
    )


class HubTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.hub = WorkerHub(
            on_event=lambda job, event: self.events.append((job, event)),
            local=SimpleNamespace(last_exit=None, close=AsyncMock()),
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

    async def connect(self, name, *, workers=WORKERS, store=None, replies=None):
        socket = FakeSocket(replies={"submit": queued, **(replies or {})})
        socket.push(hello(name, workers))
        socket.push({"state": "models", "models": store or models()})
        socket.push({"state": "ready", "protocol": 3, "pid": 1})
        task = asyncio.create_task(self.hub.serve(socket))
        task.socket = socket
        self.tasks.append(task)
        for _ in range(100):
            machine = self.hub.machines.get(name)
            if machine is not None and machine.ready.is_set():
                return socket, machine
            await asyncio.sleep(0)
        self.fail(f"{name} did not connect")

    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0)


class AuthorizationTests(unittest.TestCase):
    def test_token_must_be_configured_and_match(self):
        socket = SimpleNamespace(headers={"authorization": "Bearer secret"})
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(authorized(socket))
        with patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": "other"}):
            self.assertFalse(authorized(socket))
        with patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": "secret"}):
            self.assertTrue(authorized(socket))
            self.assertFalse(authorized(SimpleNamespace(headers={})))

    def test_requested_model_reads_either_field(self):
        video = InferenceTask(input={"feature": "video", "prompt": "sea"})
        self.assertEqual(requested_model(video), "h3-fl2va-int8-turbo")
        self.assertIsNone(requested_model(chat()))


class RegistrationTests(HubTestCase):
    async def test_unsupported_protocol_and_duplicates_are_rejected(self):
        old = FakeSocket()
        old.push(hello("box", protocol=2))
        await self.hub.serve(old)
        self.assertEqual(old.closed, 1008)
        self.assertNotIn("box", self.hub.machines)
        await self.connect("box")
        duplicate = FakeSocket()
        duplicate.push(hello("box"))
        await self.hub.serve(duplicate)
        self.assertEqual(duplicate.closed, 1008)

    async def test_first_machine_is_primary_for_model_status(self):
        await self.connect("box", store=models(tier="go"))
        self.assertEqual(self.hub.primary, "box")
        self.assertEqual(self.hub.models.selected_tier, "go")
        self.assertTrue(self.hub.connected.is_set())

    async def test_late_machine_receives_the_primary_settings(self):
        await self.connect("box", store=models(tier="pro", whisper="medium"))
        socket, _ = await self.connect("remote", store=models(tier="go"))
        configured = [message["input"] for message in socket.commands("submit")]
        self.assertIn({"feature": "tier_selection", "tier": "pro"}, configured)
        self.assertIn({"feature": "whisper_selection", "model": "medium"}, configured)


class JobTests(HubTestCase):
    async def test_job_runs_on_its_worker_and_resolves(self):
        socket, machine = await self.connect("box")
        job = await self.hub.submit(chat())
        self.assertEqual(job.worker, "llm")
        self.assertEqual(machine.ledger.workers["llm"].pending, 1)
        memory = {"capacity": [1, 2], "used": [0, 1], "residents": 1}
        socket.push({"id": job.request.id, "state": "running", "worker": "llm"})
        socket.push(
            {
                "id": job.request.id,
                "state": "succeeded",
                "worker": "llm",
                "result": {"text": "hello"},
                "memory": memory,
            }
        )
        self.assertEqual(await job.result, {"text": "hello"})
        self.assertEqual(machine.ledger.workers["llm"].pending, 0)
        self.assertFalse(machine.ledger.workers["llm"].busy)
        self.assertEqual(self.hub.memory.used, [0, 1])
        self.assertEqual(
            [event.state for _, event in self.events],
            ["queued", "running", "succeeded"],
        )

    async def test_rejection_before_acceptance_fails_submission(self):
        await self.connect(
            "box",
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
        socket, _ = await self.connect("box")
        job = await self.hub.submit(chat(), streaming=True)
        socket.push({"id": job.request.id, "state": "content", "content": "hi"})
        await self.settle()
        self.assertEqual(await job.events.get(), {"content": "hi"})

    async def test_no_machine_for_the_worker_is_unavailable(self):
        await self.connect("box", workers=["store", "image"])
        with self.assertRaises(InferenceFailure) as raised:
            await self.hub.submit(chat())
        self.assertEqual(raised.exception.status_code, 503)

    async def test_settings_go_to_every_machine_and_settle_once_all_finish(self):
        first, _ = await self.connect("box")
        second, _ = await self.connect("remote")
        job = await self.hub.submit(
            InferenceTask(
                operation="configure", input={"feature": "tier_selection", "tier": "go"}
            )
        )
        self.assertIsNone(job.worker)
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

    async def test_cancel_reaches_the_machine(self):
        socket, _ = await self.connect("box")
        job = await self.hub.submit(chat())
        await self.hub.cancel(job.request.id)
        self.assertEqual(
            socket.commands("cancel"), [{"id": job.request.id, "command": "cancel"}]
        )


class DisconnectTests(HubTestCase):
    async def test_waiting_job_moves_and_running_job_fails(self):
        first, _ = await self.connect("box")
        waiting = await self.hub.submit(chat())
        running = await self.hub.submit(chat())
        first.push({"id": running.request.id, "state": "running"})
        await self.settle()
        second, remote = await self.connect("remote")
        await first.close()
        await self.settle()
        with self.assertRaises(InferenceFailure) as raised:
            await running.result
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            [m["id"] for m in second.commands("submit")], [waiting.request.id]
        )
        self.assertEqual(waiting.machines, {"remote"})
        self.assertEqual(remote.ledger.workers["llm"].pending, 1)
        self.assertEqual(self.hub.primary, "remote")

    async def test_cancelling_a_job_without_machines_does_not_wait(self):
        await self.connect("box", replies={"submit": queued, "cancel": lambda m: []})
        job = await self.hub.submit(chat())
        job.machines = {"gone"}
        await asyncio.wait_for(self.hub.cancel_and_join(job), 1)
        with self.assertRaises(InferenceFailure):
            await job.result

    async def test_pending_controls_fail_on_disconnect(self):
        socket, machine = await self.connect("box")
        control = asyncio.create_task(machine.control("status"))
        await self.settle()
        await socket.close()
        with self.assertRaises(InferenceFailure):
            await control


class SchedulingTests(HubTestCase):
    def replies(self, loads):
        def requirement(message):
            return [
                {
                    "id": message["id"],
                    "state": "requirement",
                    "worker": message["worker"],
                    "model": "m",
                    "minimum": [GIB, 4 * GIB],
                    "preferred": [2 * GIB, 30 * GIB],
                }
            ]

        def load(message):
            loads.append((message["worker"], message["budget"]))
            return [{"id": message["id"], "state": "control", "ok": True}]

        return {"requirement": requirement, "load": load}

    async def test_waiting_worker_gets_a_budget(self):
        loads = []
        _, machine = await self.connect("box", replies=self.replies(loads))
        await self.hub.submit(chat())
        await self.hub._schedule(machine)
        slot = machine.ledger.workers["llm"]
        self.assertTrue(slot.loaded)
        self.assertEqual(loads, [("llm", [2 * GIB, 24 * GIB])])
        self.assertEqual(slot.budget, list(loads[0][1]))

    async def test_busy_reply_leaves_the_ledger_unchanged(self):
        socket, machine = await self.connect(
            "box",
            replies={
                "offload": lambda m: [
                    {"id": m["id"], "state": "control", "ok": False, "error": "busy"}
                ]
            },
        )
        image = machine.ledger.workers["image"]
        image.loaded, image.budget = True, [GIB, 24 * GIB]
        llm = machine.ledger.workers["llm"]
        llm.pending = 1
        llm.requirement = Requirement("m", (GIB, 10 * GIB), (GIB, 10 * GIB))
        await self.hub._schedule(machine)
        self.assertTrue(image.loaded)
        self.assertFalse(llm.loaded)
        self.assertEqual(len(socket.commands("offload")), 1)

    async def test_failed_requirement_still_lets_preparation_report(self):
        loads = []
        _, machine = await self.connect(
            "box",
            replies={
                "requirement": lambda m: [
                    {
                        "id": m["id"],
                        "state": "rejected",
                        "error": "no tier",
                        "status_code": 409,
                    }
                ],
                "load": self.replies(loads)["load"],
            },
        )
        await self.hub.submit(chat())
        with self.assertLogs("api.services.worker_hub", level="INFO"):
            await self.hub._schedule(machine)
        self.assertEqual(loads, [("llm", [0, 0])])

    async def test_job_that_can_never_fit_fails_instead_of_waiting(self):
        socket, machine = await self.connect(
            "box",
            replies={
                "requirement": lambda m: [
                    {
                        "id": m["id"],
                        "state": "requirement",
                        "worker": "llm",
                        "model": "huge",
                        "minimum": [GIB, 200 * GIB],
                        "preferred": [GIB, 200 * GIB],
                    }
                ]
            },
        )
        job = await self.hub.submit(chat())
        with self.assertLogs("api.services.worker_hub", level="WARNING"):
            await self.hub._schedule(machine)
        with self.assertRaises(InferenceFailure) as raised:
            await job.result
        self.assertEqual(raised.exception.status_code, 507)
        self.assertIn("huge needs at least", str(raised.exception))
        self.assertEqual(socket.commands("cancel")[0]["id"], job.request.id)
        self.assertEqual(machine.ledger.workers["llm"].pending, 0)
        self.assertFalse(socket.commands("load"))

    async def test_worker_events_update_the_ledger(self):
        socket, machine = await self.connect("box")
        socket.push(
            {
                "state": "worker",
                "event": "offloaded",
                "worker": "image",
                "loaded": False,
                "budget": [0, 0],
            }
        )
        machine.ledger.workers["image"].loaded = True
        await self.settle()
        self.assertFalse(machine.ledger.workers["image"].loaded)

    async def test_unload_offloads_on_every_machine(self):
        _, machine = await self.connect(
            "box",
            replies={
                "offload": lambda m: [{"id": m["id"], "state": "control", "ok": True}]
            },
        )
        machine.ledger.workers["llm"].loaded = True
        machine.ledger.workers["llm"].budget = [GIB, GIB]
        await self.hub.offload("llm")
        self.assertFalse(machine.ledger.workers["llm"].loaded)
        self.assertEqual(machine.ledger.workers["llm"].budget, [0, 0])
