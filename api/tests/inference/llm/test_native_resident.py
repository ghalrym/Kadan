"""Real subprocess protocol-v2 lifecycle; synthetic payloads, no CUDA or weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.inference.llm.context import ContextLimitError, ContextMemoryError
from api.inference.llm.native import HEADROOM_BYTES, PYTHON_HOST_BYTES, NativeProtocolError, WorkerProcess
from api.inference.llm.native_resident import ResidentAdapter, RESIDENT_HOST_BYTES, CACHE_RAM_BYTES
from api.inference.resources import ResourceExhausted, ResourceManager
from api.tests.inference.llm.test_native import Tokenizer, Stream

WORKER = r'''
import json, sys, time, uuid
from pathlib import Path
root=Path(sys.argv[2]); mode=lambda:(root/'mode').read_text()
if sys.argv[1]=='--plan-resident':
 if mode()=='planhang':time.sleep(30)
 print(f'plan 3 {514*1024**2} 1024 16 {sys.argv[3]} 64 {256*1024**2} 1234',flush=True);sys.exit(0)
(root/'serve-args').write_text(json.dumps(sys.argv))
session=uuid.uuid4().hex; capacity=int(sys.argv[4])
if mode()=='readyhang':time.sleep(30)
if mode()=='badready':print('ready 2 bad 16 512',flush=True)
else:print(f'ready 2 {session} 16 {capacity}',flush=True)
current=progress=0
for line in sys.stdin:
 with (root/'commands').open('a') as log:log.write(line)
 args=line.split(); command, incarnation, ident=args[:3]; fault=mode()
 assert incarnation==session
 reply_session='0'*32 if fault=='stale-session' else session
 reply_id='999' if fault=='wrong-id' else ident
 if command=='submit':
  current+=1
  print(f'queued {reply_session} {1 if fault=="reused-id" else current}',flush=True)
 elif command=='start':
  if fault=='slowstart':time.sleep(.5)
  if fault=='starthang':time.sleep(30)
  progress=0;print(f'started {reply_session} {reply_id}',flush=True)
 elif command=='step':
  if fault=='hang':time.sleep(30)
  progress+=1; token=int(args[3]); selected={7:3,3:4,4:15}.get(token,1)
  print(f'token {reply_session} {reply_id} {selected} {int(selected==15)} {progress}',flush=True)
 elif command=='end': print(f'ended {reply_session} {reply_id}',flush=True)
 elif command=='park':
  if fault=='park-failure':print('error cleanup',flush=True)
  else:print(f'parked {reply_session} {reply_id}',flush=True)
 elif command=='cache':
  retained=int(sys.argv[8])+1 if fault=='badcache' else int(sys.argv[8])//2
  print(f'cache {reply_session} {reply_id} {sys.argv[8]} {retained} {sys.argv[8]} 7 3 90 100 0 1234',flush=True)
 elif command=='close':print(f'closed {session} 0',flush=True);sys.exit(0)
'''


def fixture(root, resources):
    (root/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe','text_config':{'max_position_embeddings':512}}))
    (root/'mode').write_text('normal')
    binary=root/'worker';binary.write_text('#!'+sys.executable+'\n'+WORKER);binary.chmod(0o700)
    return ResidentAdapter(SimpleNamespace(id='small'), root, resources,
        tokenizer_factory=lambda _:Tokenizer(), streamer_factory=Stream,
        worker_path=str(binary), load_timeout=2, step_timeout=.3)


class ResidentAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.resources=ResourceManager(2*1024**3,{0:HEADROOM_BYTES+4096})
        self.adapter=fixture(self.root,self.resources)
        self.addCleanup(self.adapter.close)
        self.adapter.configure_context(512)

    def generate(self, **kwargs):
        return self.adapter.generate([{'role':'user','text':'Hi'}], **kwargs)

    def mode(self, value):
        (self.root/'mode').write_text(value)

    def test_consecutive_requests_and_park_restore_keep_process_and_backing_budget(self):
        worker=self.adapter.worker;session=self.adapter.session;host=self.adapter.host
        self.assertEqual(self.generate(),'AB');self.assertEqual(self.generate(),'AB')
        self.resources.offload_workload_devices('llm')
        self.assertTrue(self.adapter.worker_alive);self.assertFalse(self.adapter.is_resident)
        self.assertIs(self.adapter.host,host)
        rows=self.resources.snapshot()['reservations']
        self.assertEqual(sum(sum(row['device_bytes'].values()) for row in rows.values()),HEADROOM_BYTES)
        self.assertEqual(sum(row['host_bytes'] for row in rows.values()),RESIDENT_HOST_BYTES+CACHE_RAM_BYTES+PYTHON_HOST_BYTES)
        self.assertEqual(self.generate(),'AB')
        self.assertIs(self.adapter.worker,worker);self.assertEqual(self.adapter.session,session)
        commands=[line.split() for line in (self.root/'commands').read_text().splitlines()]
        self.assertEqual([row[2] for row in commands if row[0]=='start'],['1','2','3'])
        self.assertEqual(sum(row[0]=='end' for row in commands),3)
        self.assertEqual(sum(row[0]=='park' for row in commands),1)
        args=json.loads((self.root/'serve-args').read_text())
        self.assertEqual(int(args[5]),RESIDENT_HOST_BYTES+CACHE_RAM_BYTES)
        self.assertEqual(int(args[6]),HEADROOM_BYTES+1024)
        self.assertEqual(int(args[8]),CACHE_RAM_BYTES)
        self.adapter.close();self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_parked_context_cannot_be_spent_and_restoration_pressure_is_retryable(self):
        self.adapter.offload_to_ram()
        # Active external reservation leaves too little for restore; native
        # context remains protected and no physical command has been sent.
        other=self.resources.reserve('image','image',device_bytes={0:4096})
        try:
            with self.assertRaises(ContextMemoryError):self.generate()
            self.assertTrue(self.adapter.worker_alive);self.assertTrue(self.adapter.parked)
        finally:other.release()
        self.assertEqual(self.generate(),'AB')

    def test_invalid_prompt_does_not_submit(self):
        self.adapter.tokenizer.tokens=[2]*400
        with self.assertRaises(ContextLimitError):self.generate()
        self.assertNotIn('submit ',(self.root/'commands').read_text())

    def test_stale_session_wrong_id_and_reused_id_reap_failed_child(self):
        for mode in ('stale-session','wrong-id','reused-id'):
            with self.subTest(mode=mode):
                if not self.adapter.worker_alive:
                    self.adapter._close_locked();self.mode('normal');self.adapter.configure_context(512)
                self.generate();self.mode(mode)
                with self.assertRaises(NativeProtocolError):self.generate()
                self.assertFalse(self.adapter.worker_alive)

    def test_new_process_has_new_session_and_restarts_request_ids(self):
        self.generate();session=self.adapter.session
        self.adapter._evict();self.generate()
        self.assertNotEqual(self.adapter.session,session);self.assertEqual(self.adapter.last_id,1)

    def test_active_cancellation_terminates_child_before_release(self):
        self.mode('hang');cancel=threading.Event();timer=threading.Timer(.05,cancel.set);timer.start()
        try:
            with self.assertRaises(InterruptedError):self.generate(cancel_event=cancel)
        finally:timer.join()
        self.assertFalse(self.adapter.worker_alive)
        self.adapter.close();self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_unconfirmed_park_blocks_reuse_and_retains_all_envelopes(self):
        self.mode('park-failure');before=self.resources.snapshot()['reservations']
        with self.assertRaises(NativeProtocolError):self.adapter.offload_to_ram()
        self.assertEqual(self.resources.snapshot()['reservations'],before)
        with self.assertRaises(NativeProtocolError):self.generate()
        with patch.object(self.adapter.worker,'stop',side_effect=subprocess.TimeoutExpired('child',5)):
            with self.assertRaises(subprocess.TimeoutExpired):self.adapter._evict()
        self.assertEqual(self.resources.snapshot()['reservations'],before)
        self.adapter._evict();self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_startup_timeouts_and_bad_readiness_reconcile_all_three_budgets(self):
        self.adapter.load_timeout=.1
        for mode in ('planhang','readyhang','badready'):
            with self.subTest(mode=mode):
                self.adapter._evict();self.mode(mode)
                with self.assertRaises((TimeoutError,NativeProtocolError)):
                    self.adapter.configure_context(512)
                self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_failed_startup_reap_retains_context_host_and_arena(self):
        self.adapter._evict();self.mode('badready')
        stop=WorkerProcess.stop
        def fail_stop(worker):
            if worker.process is not None and '--serve-resident' in worker.process.args:
                raise subprocess.TimeoutExpired('child',5)
            return stop(worker)
        with patch.object(WorkerProcess,'stop',fail_stop):
            with self.assertRaises(subprocess.TimeoutExpired):self.adapter.configure_context(512)
        self.assertEqual(len(self.resources.snapshot()['reservations']),4)
        self.adapter._evict();self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_restore_uses_load_deadline_and_keeps_step_deadline(self):
        self.adapter.offload_to_ram(); self.mode('slowstart')
        self.assertEqual(self.generate(), 'AB')  # .5s start exceeds .3s step limit.
        stats=self.adapter.cache_stats()
        self.assertEqual(stats['retained_bytes'], CACHE_RAM_BYTES//2)
        self.assertEqual((stats['hits'],stats['source_bytes']),(7,100))
        self.mode('hang')
        with self.assertRaises(TimeoutError): self.generate()
        self.adapter.close(); self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_restore_timeout_and_cancellation_reap_before_release(self):
        self.adapter.offload_to_ram(); self.adapter.load_timeout=.1; self.mode('starthang')
        with self.assertRaises(TimeoutError): self.generate()
        self.adapter.close(); self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_complete_cache_selected_and_explicit_cap_honored(self):
        self.assertEqual(self.adapter.cache_capacity,self.adapter.packed_weight_bytes)
        self.adapter._evict(); self.adapter.requested_cache_bytes=1024
        self.adapter.configure_context(512)
        self.assertEqual(self.adapter.cache_capacity,1024)
        self.assertEqual(self.adapter.cache_stats()['capacity_bytes'],1024)

    def test_cache_report_outside_admission_quarantines_before_release(self):
        self.mode('badcache'); before=self.resources.snapshot()['reservations']
        with self.assertRaises(NativeProtocolError): self.adapter.offload_to_ram()
        self.assertTrue(self.adapter.quarantined)
        self.assertEqual(self.resources.snapshot()['reservations'],before)
        self.adapter.close(); self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_cancel_during_restore_reaps_child(self):
        self.adapter.offload_to_ram(); self.mode('starthang')
        cancel=threading.Event(); timer=threading.Timer(.05,cancel.set); timer.start()
        try:
            with self.assertRaises(InterruptedError): self.generate(cancel_event=cancel)
        finally: timer.join()
        self.assertFalse(self.adapter.worker_alive)
        self.adapter.close(); self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_cache_snapshot_cleared_only_after_confirmed_reap(self):
        stats=self.adapter.cache_stats()
        self.assertGreater(stats['retained_bytes'],0)
        with patch.object(self.adapter.worker,'stop',side_effect=subprocess.TimeoutExpired('child',5)):
            with self.assertRaises(subprocess.TimeoutExpired):self.adapter._evict()
        self.assertEqual(self.adapter.cache_stats(),stats)
        self.adapter.close()
        self.assertIsNone(self.adapter.cache_stats())
