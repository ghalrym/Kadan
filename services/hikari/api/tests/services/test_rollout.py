import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from kadan_tracing.rollout import Admission

from api.services.inference_errors import InferenceFailure
from api.services.rollout import WORKER_KINDS, NativeBarrier
from api.services.worker_hub import Computer, Worker, WorkerHub


def fake_cohort(tmp_path):
    fence = tmp_path / "fence.json"
    fence.write_text('{"closed":[]}')
    gate = Admission("hikari-api", fence)
    hub = WorkerHub()
    for kind in WORKER_KINDS:
        ready = asyncio.Event()
        ready.set()
        hub.workers[kind] = SimpleNamespace(
            kind=kind,
            instance=f"container-{kind}",
            generation=str(uuid4()),
            rollout_generation="",
            ready=ready,
            preparing=None,
            controls={},
            control=AsyncMock(
                return_value={
                    "state": "status",
                    "workers": [
                        {"worker": lane, "busy": False, "queued": 0, "releasing": False}
                        for lane in ("store", kind)
                    ],
                }
            ),
        )
    barrier = NativeBarrier(hub, gate, lambda: gate.active + hub.submitting)
    return (hub, gate, barrier)


def close_gate(gate):
    gate.fence.write_text('{"closed":["hikari-api"]}')


class NativeBarrierTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        environment = patch.dict(os.environ, {"KADAN_ROLLOUT_GENERATION": ""})
        environment.start()
        self.addCleanup(environment.stop)

    async def test_idle_barrier_queries_every_lane_and_freezes_maintenance(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        proof = await barrier.prove()
        self.assertTrue(proof["native_idle"])
        self.assertEqual({worker["kind"] for worker in proof["workers"]}, WORKER_KINDS)
        self.assertTrue(hub.rollout_paused())
        self.assertTrue(barrier.status()["native_idle"])
        for worker in hub.workers.values():
            worker.control.assert_awaited_once_with("status", timeout=5)
        gate.fence.write_text('{"closed":[]}')
        self.assertFalse(hub.rollout_paused())
        self.assertFalse(barrier.status()["native_idle"])

    async def test_barrier_rejects_busy_store_and_inference_lanes(self):
        tmp_path = self.path
        for lane in [0, 1]:
            with self.subTest(lane=lane):
                for field, value in [
                    ("busy", True),
                    ("queued", 1),
                    ("releasing", True),
                ]:
                    with self.subTest(field=field, value=value):
                        hub, gate, barrier = fake_cohort(tmp_path)
                        worker = next(iter(hub.workers.values()))
                        worker.control.return_value["workers"][lane][field] = value
                        close_gate(gate)
                        with self.assertRaisesRegex(InferenceFailure, "not idle"):
                            await barrier.prove()
                        self.assertTrue(hub.rollout_paused())
                        self.assertFalse(barrier.status()["native_idle"])

    async def test_barrier_rejects_incomplete_or_malformed_evidence(self):
        tmp_path = self.path
        for fault in [
            "missing_lane",
            "invalid_count",
            "missing_field",
            "duplicate_kind",
        ]:
            with self.subTest(fault=fault):
                hub, gate, barrier = fake_cohort(tmp_path)
                worker = next(iter(hub.workers.values()))
                lanes = worker.control.return_value["workers"]
                if fault == "missing_lane":
                    lanes.pop()
                elif fault == "invalid_count":
                    lanes[0]["queued"] = "0"
                elif fault == "missing_field":
                    del lanes[0]["busy"]
                else:
                    lanes[1]["worker"] = "store"
                close_gate(gate)
                with self.assertRaises(InferenceFailure):
                    await barrier.prove()
                self.assertFalse(barrier.status()["native_idle"])

    async def test_cohort_change_during_status_invalidates_proof(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        worker = next(iter(hub.workers.values()))
        reply = worker.control.return_value

        async def changed(*args, **kwargs):
            hub.rollout_epoch += 1
            return reply

        worker.control.side_effect = changed
        close_gate(gate)
        with self.assertRaisesRegex(InferenceFailure, "changed"):
            await barrier.prove()
        self.assertFalse(barrier.proof_valid())

    async def test_barrier_requires_no_http_work_or_pending_submission(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        for field in ("active", "submitting"):
            target = gate if field == "active" else hub
            setattr(target, field, 1)
            with self.assertRaisesRegex(InferenceFailure, "drained"):
                await barrier.prove()
            setattr(target, field, 0)
        self.assertFalse(hub.rollout_frozen)

    async def test_barrier_serializes_with_maintenance_and_rechecks_work(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        await hub.maintenance.acquire()
        proof = asyncio.create_task(barrier.prove())
        await asyncio.sleep(0)
        self.assertFalse(proof.done())
        gate.active = 1
        hub.maintenance.release()
        with self.assertRaisesRegex(InferenceFailure, "changed"):
            await proof
        self.assertFalse(hub.rollout_frozen)

    async def test_uncertain_disconnect_or_pending_control_cannot_form_barrier(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        hub.rollout_safe = False
        with self.assertRaisesRegex(InferenceFailure, "uncertain"):
            await barrier.prove()
        hub.rollout_safe = True
        next(iter(hub.workers.values())).controls["pending"] = object()
        with self.assertRaisesRegex(InferenceFailure, "in flight"):
            await barrier.prove()
        self.assertFalse(barrier.proof_valid())

    async def test_cold_api_requires_host_token_and_fresh_worker_generations(self):
        tmp_path = self.path
        hub, gate, previous = fake_cohort(tmp_path)
        close_gate(gate)
        prior_proof = await previous.prove()
        token = str(uuid4())
        marker = gate.fence.with_name("native.json")
        marker.write_text(
            json.dumps(
                {
                    "phase": "cold_start",
                    "generation": token,
                    "old_workers": prior_proof["workers"],
                }
            )
        )
        cold = NativeBarrier(hub, gate, lambda: gate.active)
        self.assertFalse(cold.ready())
        with self.assertRaisesRegex(InferenceFailure, "attestation"):
            await cold.prove()
        with patch.dict(os.environ, {"KADAN_ROLLOUT_GENERATION": token}):
            cold = NativeBarrier(hub, gate, lambda: gate.active)
            with self.assertRaisesRegex(InferenceFailure, "survived"):
                await cold.prove()
            for worker in hub.workers.values():
                worker.generation = str(uuid4())
                worker.rollout_generation = token
            self.assertFalse(cold.ready())
            proof = await cold.prove()
            self.assertEqual(proof["cold_start_generation"], token)
            self.assertTrue(cold.ready())
            gate.fence.write_text('{"closed":[]}')
            self.assertTrue(cold.ready())
            self.assertFalse(cold.proof_valid())

    async def test_socket_send_rechecks_freeze_after_transport_lock(self):
        tmp_path = self.path
        hub, gate, _ = fake_cohort(tmp_path)
        socket = SimpleNamespace(send_text=AsyncMock())
        computer = Computer({"id": "machine", "name": "machine", "ram_free": 1})
        worker = Worker(
            socket,
            {
                "worker": "llm",
                "instance": "one",
                "generation": str(uuid4()),
                "rollout_generation": "",
                "machine": {"devices": []},
            },
            computer,
        )
        worker.hub = hub
        await worker.write_lock.acquire()
        sending = asyncio.create_task(worker.send({"id": "request", "command": "load"}))
        await asyncio.sleep(0)
        close_gate(gate)
        hub.rollout_frozen = True
        worker.write_lock.release()
        with self.assertRaisesRegex(InferenceFailure, "frozen"):
            await sending
        socket.send_text.assert_not_awaited()
        await worker.send({"id": "request", "command": "status"})
        socket.send_text.assert_awaited_once()

    async def test_postproof_membership_change_revokes_idle_evidence(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        await barrier.prove()
        self.assertTrue(barrier.proof_valid())
        hub.rollout_epoch += 1
        self.assertFalse(barrier.proof_valid())
        self.assertEqual(barrier.status()["workers"], [])

    async def test_offload_cannot_start_after_idle_barrier(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        await barrier.prove()
        with self.assertRaisesRegex(InferenceFailure, "frozen"):
            await hub.offload("llm")

    async def test_inflight_preparation_prevents_idle_barrier(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        release = asyncio.Event()
        task = asyncio.create_task(release.wait())
        next(iter(hub.workers.values())).preparing = task
        with self.assertRaisesRegex(InferenceFailure, "complete ready"):
            await barrier.prove()
        release.set()
        await task

    async def test_frozen_worker_ready_event_is_rejected(self):
        tmp_path = self.path
        hub, gate, barrier = fake_cohort(tmp_path)
        close_gate(gate)
        await barrier.prove()
        worker = next(iter(hub.workers.values()))
        worker.websocket = SimpleNamespace(close=AsyncMock())
        await hub._dispatch(worker, {"state": "ready"})
        worker.websocket.close.assert_awaited_once_with(code=1013)
        self.assertFalse(barrier.proof_valid())
        self.assertFalse(barrier.ready())
