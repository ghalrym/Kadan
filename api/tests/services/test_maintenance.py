import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.services.maintenance import Admission, MaintenanceBusy
from api.services.model_downloads import ModelManager
from api.services.runtime import RuntimeManager
from api.routes.internal import updates


class AdmissionTests(unittest.TestCase):
    def test_seal_prevents_new_work_but_preserves_existing_owner(self):
        gate = Admission()
        ticket = gate.enter()
        gate.seal()
        with self.assertRaises(MaintenanceBusy):
            gate.enter()
        self.assertEqual(gate.count(), 1)
        ticket.close()
        ticket.close()
        self.assertEqual(gate.count(), 0)
        gate.resume()
        gate.enter().close()

    def test_restart_marker_blocks_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'maintenance'
            marker.touch()
            with patch.dict('os.environ', KADAN_MAINTENANCE_FILE=str(marker)):
                with self.assertRaises(MaintenanceBusy):
                    Admission().enter()

    def test_download_status_completion_does_not_release_cleanup_ticket(self):
        gate, entered, finish = Admission(), threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            manager = ModelManager(Path(folder))
            def download(entry, lease):
                try:
                    manager._jobs[entry.id].status = 'complete'
                    entered.set()
                    finish.wait(2)  # staging removal and writer lease still owned
                finally:
                    lease.close()
            with patch('api.services.model_downloads.admission', gate), patch.object(manager, '_download', side_effect=download):
                manager.start('small')
                try:
                    self.assertTrue(entered.wait(2))
                    gate.seal()
                    self.assertEqual(manager._jobs['small'].status, 'complete')
                    self.assertEqual(gate.count(), 1)
                finally:
                    finish.set()
                    manager._thread.join(2)
                self.assertEqual(gate.count(), 0)

    def test_download_start_failure_releases_ticket(self):
        gate = Admission()
        with tempfile.TemporaryDirectory() as folder, patch('api.services.model_downloads.admission', gate), \
             patch('api.services.model_downloads.threading.Thread.start', side_effect=RuntimeError('thread failed')):
            with self.assertRaises(RuntimeError):
                ModelManager(Path(folder)).start('small')
            self.assertEqual(gate.count(), 0)


class DrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_load_ownership_outlives_accepted_response(self):
        gate, finish = Admission(), asyncio.Event()
        runtime = RuntimeManager()
        async def load():
            await finish.wait()
        def start(manager):
            runtime.task = asyncio.create_task(load())
            return {'state': 'loading'}
        with patch('api.services.runtime.admission', gate), patch.object(runtime, '_start_load_owned', side_effect=start):
            self.assertEqual(runtime._start_load(None), {'state': 'loading'})
            gate.seal()
            self.assertEqual(gate.count(), 1)
            finish.set()
            await runtime.task
            await asyncio.sleep(0)
            self.assertEqual(gate.count(), 0)

    async def test_waits_for_real_work_then_cleanup_before_drained(self):
        gate, finish = Admission(), asyncio.Event()
        ticket = gate.enter()
        decision = Mock(close=AsyncMock(side_effect=finish.wait))
        runtime = Mock(close=AsyncMock())
        with patch.object(updates, 'admission', gate), patch.object(updates, 'decision_manager', decision), \
             patch.object(updates, 'runtime_manager', runtime), patch.object(updates, 'model_manager', Mock()), \
             patch.object(updates, 'cancel', False):
            gate.seal()
            task = asyncio.create_task(updates.drain_work(timeout=1))
            await asyncio.sleep(.06)
            decision.close.assert_not_awaited()
            ticket.close()
            await asyncio.sleep(.06)
            self.assertEqual(updates.state, 'cleaning')
            self.assertFalse(task.done())
            finish.set()
            await task
            self.assertEqual(updates.state, 'drained')
            runtime.close.assert_awaited_once()

    async def test_timeout_leaves_work_running_without_cleanup_or_kill(self):
        gate = Admission()
        ticket = gate.enter()
        decision = Mock(close=AsyncMock())
        with patch.object(updates, 'admission', gate), patch.object(updates, 'decision_manager', decision), \
             patch.object(updates, 'cancel', False):
            await updates.drain_work(timeout=.001)
            self.assertEqual(updates.state, 'busy')
            self.assertEqual(gate.count(), 1)
            decision.close.assert_not_awaited()
        ticket.close()

    async def test_cleanup_failure_does_not_claim_drained(self):
        with patch.object(updates, 'admission', Admission()), patch.object(updates, 'cancel', False), \
             patch.object(updates, 'decision_manager', Mock(close=AsyncMock(side_effect=RuntimeError('still owns memory')))):
            await updates.drain_work()
            self.assertEqual(updates.state, 'error')
