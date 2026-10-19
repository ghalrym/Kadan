import asyncio
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi import WebSocketDisconnect

from api.pydantic_models.inference_task import InferenceTask
from api.services.inference_errors import InferenceFailure
from api.services.memory_scheduler import GIB, Requirement
from api.services.worker_hub import (
    RELEASE_TIMEOUT,
    WorkerHub,
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


def hello(kind, *, computer="boot-1", instance=None, gpus=(("GPU-0", 24),), protocol=5):
    return {
        "state": "hello",
        "protocol": protocol,
        "pid": 1,
        "generation": str(uuid4()),
        "rollout_generation": "",
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


def models(tier=None, whisper=None, checkpoints=()):
    return {
        "models": list(checkpoints),
        "selected_tier": tier,
        "selected_whisper": whisper,
        "whisper_formatting": True,
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
        socket.push({"state": "ready", "protocol": 5, "pid": 1})
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
    async def test_unsupported_protocol_and_unknown_types_are_rejected(self):
        for message in (hello("llm", protocol=3), hello("gpu")):
            socket = FakeSocket()
            socket.push(message)
            await self.hub.serve(socket)
            self.assertEqual(socket.closed, 1008)
        self.assertFalse(self.hub.workers)

    async def test_workers_on_one_computer_share_its_ram_and_each_gpu_once(self):
        _, language = await self.connect("llm", gpus=(("GPU-a", 24),))
        _, picture = await self.connect("image", gpus=(("GPU-a", 24),))
        self.assertIs(language.computer, picture.computer)
        ledger = language.computer.ledger
        self.assertEqual(ledger.capacity, [64 * GIB, 24 * GIB])
        self.assertEqual(set(ledger.workers), {language.id, picture.id})

    async def test_workers_on_other_computers_have_their_own_ledgers(self):
        _, here = await self.connect("llm", computer="boot-1")
        _, there = await self.connect("llm", computer="boot-2", instance="llm-2")
        self.assertIsNot(here.computer, there.computer)
        self.assertEqual(len(self.hub.computers), 2)

    async def test_a_worker_that_sees_some_gpus_gets_budgets_in_its_own_order(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        # The language worker sees both GPUs and will start on GPU-a.
        await self.connect(
            "llm",
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

    async def test_computer_ledger_goes_when_its_last_worker_leaves(self):
        socket, worker = await self.connect("llm")
        await socket.close()
        await self.settle()
        self.assertNotIn(worker.computer.id, self.hub.computers)

    async def test_first_worker_is_primary_for_model_status(self):
        await self.connect("llm", store=models(tier="go"))
        self.assertEqual(self.hub.models.selected_tier, "go")

    async def test_late_worker_receives_the_primary_settings(self):
        primary = models(
            tier="pro",
            whisper="medium",
            checkpoints=[checkpoint("qwen", context_limit=32768)],
        )
        await self.connect("llm", store=primary)
        late = models(tier="go", checkpoints=[checkpoint("qwen", context_limit=65536)])
        socket, _ = await self.connect(
            "image", computer="boot-2", store=late, replies={"submit": configured}
        )
        sent = [message["input"] for message in socket.commands("submit")]
        self.assertIn({"feature": "tier_selection", "tier": "pro"}, sent)
        self.assertIn({"feature": "whisper_selection", "model": "medium"}, sent)
        self.assertIn(
            {"feature": "model_context", "model_id": "qwen", "context_limit": 32768},
            sent,
        )

    async def test_late_worker_is_not_routable_until_settings_apply(self):
        await self.connect("llm", store=models(tier="pro"))
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
        socket.push({"state": "models", "models": models(tier="go")})
        socket.push({"state": "ready", "protocol": 5, "pid": 1})
        with self.assertLogs("api.services.worker_hub", level="ERROR"):
            await self.hub.serve(socket)
        self.assertEqual(socket.closed, 1011)
        self.assertEqual([w.kind for w in self.hub.workers.values()], ["llm"])


class JobTests(HubTestCase):
    async def test_job_runs_on_a_worker_of_its_type_and_resolves(self):
        await self.connect("image")
        socket, worker = await self.connect("llm")
        job = await self.hub.submit(chat())
        self.assertEqual(job.machines, {worker.id})
        self.assertEqual(worker.slot.pending, 1)
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
        self.assertEqual(worker.slot.pending, 0)
        self.assertFalse(worker.slot.busy)
        self.assertEqual(self.hub.memory.used, [0, 1])
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
            "llm",
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
        socket, _ = await self.connect("llm")
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
        _, worker = await self.connect("llm")
        job = await submission
        self.assertEqual(job.machines, {worker.id})

    async def test_settings_go_to_every_worker_and_settle_once_all_finish(self):
        first, _ = await self.connect("llm")
        second, _ = await self.connect("image")
        job = await self.hub.submit(
            InferenceTask(
                operation="configure", input={"feature": "tier_selection", "tier": "go"}
            )
        )
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
        store = models(checkpoints=[checkpoint("qwen-image", kind="image")])
        language, _ = await self.connect("llm", store=store)
        here, _ = await self.connect("image", instance="image-1", store=store)
        twin, _ = await self.connect("image", instance="image-2", store=store)
        there, _ = await self.connect("image", computer="boot-2", store=store)
        await self.hub.submit(
            InferenceTask(
                operation="download",
                input={"feature": "models", "model_id": "qwen-image"},
            )
        )
        self.assertFalse(language.commands("submit"))
        self.assertEqual(len(here.commands("submit")) + len(twin.commands("submit")), 1)
        self.assertEqual(len(there.commands("submit")), 1)

    async def test_cancel_reaches_the_worker(self):
        socket, _ = await self.connect("llm")
        job = await self.hub.submit(chat())
        await self.hub.cancel(job.request.id)
        self.assertEqual(
            socket.commands("cancel"), [{"id": job.request.id, "command": "cancel"}]
        )


class DisconnectTests(HubTestCase):
    async def test_waiting_job_moves_and_running_job_fails(self):
        first, _ = await self.connect("llm", instance="llm-1")
        waiting = await self.hub.submit(chat())
        running = await self.hub.submit(chat())
        first.push({"id": running.request.id, "state": "running"})
        await self.settle()
        second, other = await self.connect("llm", instance="llm-2")
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
        await self.connect("llm", replies={"cancel": lambda m: []})
        job = await self.hub.submit(chat())
        job.machines = {"gone"}
        await asyncio.wait_for(self.hub.cancel_and_join(job), 1)
        with self.assertRaises(InferenceFailure):
            await job.result

    async def test_disconnect_invalidates_rollout_queue_knowledge(self):
        socket, _ = await self.connect("llm")
        self.assertTrue(self.hub.rollout_safe)
        await socket.close()
        await self.settle()
        self.assertFalse(self.hub.rollout_safe)
        await self.connect("llm")
        self.assertFalse(self.hub.rollout_safe)

    async def test_pending_controls_fail_on_disconnect(self):
        socket, worker = await self.connect("llm")
        control = asyncio.create_task(worker.control("status"))
        await self.settle()
        await socket.close()
        with self.assertRaises(InferenceFailure):
            await control


class SchedulingTests(HubTestCase):
    async def test_waiting_worker_gets_a_budget(self):
        _, worker = await self.connect(
            "llm", replies={"requirement": needs((1, 4), (2, 30)), "load": ok}
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

        await self.connect("llm", replies={"requirement": needs((1, 6), (2, 30))})
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
        _, language = await self.connect("llm")
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
            "llm",
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
            "llm", replies={"requirement": needs((1, 200), (1, 200), model="huge")}
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
        await self.connect("llm", gpus=(("GPU-a", 24), ("GPU-b", 24)))
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
        socket, worker = await self.connect("llm", replies={"offload": ok})
        slot = worker.slot
        slot.loaded, slot.budget = True, [GIB, GIB]
        await self.hub.offload("llm")
        # The charge stays until the worker confirms the memory is free.
        self.assertTrue(slot.loaded and slot.releasing)
        self.assertEqual(slot.budget, [GIB, GIB])
        socket.push(
            {
                "state": "worker",
                "event": "offloaded",
                "worker": "llm",
                "loaded": False,
                "budget": [0, 0],
            }
        )
        await self.settle()
        self.assertFalse(slot.loaded or slot.releasing)
        self.assertEqual(slot.budget, [0, 0])


class ResidentRequirementTests(HubTestCase):
    async def test_routine_snapshots_keep_requirements_and_changes_drop_them(self):
        socket, worker = await self.connect("llm", store=models(tier="go"))
        worker.slot.requirement = Requirement("m", (GIB, GIB), (GIB, GIB))
        socket.push({"state": "models", "models": models(tier="go")})
        await self.settle()
        self.assertIsNotNone(worker.slot.requirement)
        socket.push({"state": "models", "models": models(tier="pro")})
        await self.settle()
        self.assertIsNone(worker.slot.requirement)

    async def test_stale_grant_is_replaced_once_the_requirement_grows(self):
        loads = []

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        _, worker = await self.connect(
            "llm", replies={"requirement": needs((1, 6), (2, 30)), "load": load}
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

        llm_socket, language = await self.connect("llm", replies={"load": load})
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
                "worker": "llm",
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
                    "workers": [{"worker": "store"}, {"worker": "llm", **status}],
                }
            ]

        _, worker = await self.connect("llm", replies={"status": reply})
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


class LoadContextTests(HubTestCase):
    async def connect_with_context_needs(self, asked, loads):
        def requirement(message):
            asked.append(
                {k: v for k, v in message.items() if k in ("model", "context")}
            )
            # Only the saved context is too large for this machine.
            device = 6 if "context" in message else 200
            return needs((1, device), (1, device), model="qwen")(message)

        def load(message):
            loads.append(message["budget"])
            return ok(message)

        return await self.connect(
            "llm", replies={"requirement": requirement, "load": load}
        )

    def load_task(self, **context):
        return InferenceTask(operation="load", input={"feature": "llm", **context})

    async def test_a_smaller_load_context_is_planned_instead_of_rejected(self):
        asked, loads = [], []
        _, worker = await self.connect_with_context_needs(asked, loads)
        job = await self.hub.submit(self.load_task(context_limit=8192))
        await self.hub._schedule(worker.computer)
        self.assertEqual(asked[-1], {"model": None, "context": 8192})
        self.assertEqual(loads, [[GIB, 6 * GIB]])
        self.assertFalse(job.result.done())

    async def test_explicit_null_and_omitted_contexts_are_distinct(self):
        asked, loads = [], []
        _, worker = await self.connect_with_context_needs(asked, loads)
        worker.slot.loaded = worker.slot.busy = True
        await self.hub.submit(self.load_task(context_limit=None))
        await self.hub._schedule(worker.computer)
        self.assertEqual(asked[-1], {"model": None, "context": None})
        await self.hub.submit(chat())
        self.assertEqual(
            self.hub._head_fields(worker), {"model": None, "context": None}
        )
        self.hub.jobs.pop(next(iter(self.hub.jobs)))
        self.assertEqual(self.hub._head_fields(worker), {"model": None})

    async def test_requirement_follows_a_new_head_job(self):
        asked, loads = [], []
        _, worker = await self.connect_with_context_needs(asked, loads)
        first = await self.hub.submit(self.load_task(context_limit=8192))
        await self.hub._schedule(worker.computer)
        self.hub.jobs.pop(first.request.id)
        worker.jobs.discard(first.request.id)
        await self.hub.submit(self.load_task(context_limit=4096))
        worker.slot.busy = True
        await self.hub._schedule(worker.computer)
        self.assertEqual(asked[-1], {"model": None, "context": 4096})
