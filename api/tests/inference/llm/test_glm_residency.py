import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from api.inference.llm.glm_residency import ResidentGlmAdapter, MIB
from api.inference.resources import ResourceManager

GIB = 1024**3

class Process:
    commands = []
    corrupt = False
    fail_ready = False
    def __init__(self):
        self.closed = False
        self.io_ready = False
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
    def start(self, command):
        self.command = command
        self.commands.append(command)
        self.io_ready = True
    def read(self, *args):
        if self.command[1] == '--plan-glm':
            context = int(self.command[3])
            host = 640*MIB + 34*(64*128*128+3*8192*4)*4 + 11*(context*512+((context+3)//4)*128+8*128)*4
            budgets = self.command[4].replace(':',' ').replace(',',' ')
            return f'plan 5 {host+int(self.corrupt)} {1200*GIB} 154880 {context} {190*GIB} 110110 2 {budgets}'
        if self.fail_ready:
            raise RuntimeError('mock startup failed')
        return 'ready 2 ' + 'b'*32 + ' 154880 ' + self.command[3]
    def exchange(self, command, *args):
        verb, session, request = command.split()
        if verb == 'cache':
            return f'cache {session} {request} 0 0 0 0 0 0 0 0 0'
        return f'{dict(park="parked", close="closed")[verb]} {session} {request}'
    def finish(self):
        pass
    def stop(self):
        self.closed = True

class GlmResidencyTests(unittest.TestCase):
    def setUp(self):
        Process.commands=[];Process.corrupt=False;Process.fail_ready=False
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        root=Path(self.directory.name)
        (root/'config.json').write_text(json.dumps({'model_type':'glm5_next','text_config':{'max_position_embeddings':1048576}}))
        worker=root/'worker';worker.touch();worker.chmod(0o700)
        self.resources=ResourceManager(8*GIB,{0:12*GIB,1:12*GIB})
        self.adapter=ResidentGlmAdapter(SimpleNamespace(id='large'),root,self.resources,
            worker_path=worker,tokenizer_factory=Mock())
        self.mock=patch('api.inference.llm.glm_residency.LineProtocolProcess',Process)
        self.mock.start();self.addCleanup(self.mock.stop)
    def test_larger_than_vram_uses_dynamic_host_budget_and_parks_all_devices(self):
        self.adapter.configure_context(1024)
        self.assertGreater(self.adapter.packed_weight_bytes,24*GIB)
        self.assertEqual(set(self.adapter.execution_bytes),{0,1})
        self.assertEqual(self.adapter.cache_capacity,0)
        self.assertTrue(self.adapter.is_resident)
        self.adapter.offload_to_ram()
        self.assertIsNone(self.adapter.reservation)
        self.adapter._restore_locked(None)
        self.assertIsNotNone(self.adapter.reservation)
        self.adapter.close()
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_plan_mismatch_reaps_and_releases(self):
        Process.corrupt=True
        with self.assertRaisesRegex(Exception,'admitted bounds'):
            self.adapter.configure_context(1024)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_readiness_failure_reaps_and_releases(self):
        Process.fail_ready=True
        with self.assertRaisesRegex(RuntimeError,'mock startup failed'):
            self.adapter.configure_context(1024)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_architecture_maximum_is_admitted_without_truncation(self):
        self.resources.capacity = type(self.resources.capacity)(64*GIB, {0:12*GIB,1:12*GIB})
        self.adapter.configure_context(None)
        self.assertEqual(self.adapter.capacity,1048576)
        self.assertEqual(Process.commands[0][3],'1048576')
        self.adapter.close()
        self.assertEqual(self.resources.snapshot()['reservations'],{})
