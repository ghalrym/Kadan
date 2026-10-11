import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from api.inference.llm.qwen_residency import ResidentQwenAdapter, HEADROOM_BYTES
from api.inference.resources import MemoryCapacity, ResourceManager

GIB = 1024**3

class Process:
    commands = []
    def __init__(self):
        self.closed = False
        self.io_ready = False
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
    def start(self, command):
        self.command = command
        self.commands.append(command)
        self.io_ready = True
    def read(self, *args):
        if self.command[1] == '--plan-auto':
            return f'plan 4 538968064 {22*GIB} 100 1024 4100 {20*GIB} 200 1 2 0 {5*GIB} 1 {6*GIB}'
        return 'ready 2 ' + 'a'*32 + ' 100 1024'
    def exchange(self, command, *args):
        verb, session, request = command.split()
        if verb == 'cache':
            return f'cache {session} {request} {20*GIB} 0 0 0 0 0 0 0 0'
        return f'{dict(park="parked", close="closed")[verb]} {session} {request}'
    def finish(self):
        pass
    def stop(self):
        self.closed = True

class AutomaticResidencyTests(unittest.TestCase):
    def test_distributed_streaming_admission_and_park_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe', 'text_config':{'max_position_embeddings':262144}}))
            worker = root/'worker';worker.write_text('');worker.chmod(0o700)
            resources = ResourceManager(64*GIB, {0:12*GIB, 1:12*GIB})
            adapter = ResidentQwenAdapter(SimpleNamespace(id='small'), root, resources,
                worker_path=worker, tokenizer_factory=Mock())
            Process.commands = []
            with patch('api.inference.llm.qwen_residency.LineProtocolProcess', Process), patch(
                'api.inference.llm.qwen_residency.probe_memory', return_value=MemoryCapacity(64*GIB)):
                adapter.configure_context(1024)
                self.assertTrue(adapter.streaming_weights)
                self.assertEqual(adapter.execution_bytes, {0:5*GIB-HEADROOM_BYTES, 1:6*GIB-HEADROOM_BYTES})
                serve = Process.commands[-1]
                self.assertEqual(serve[1], '--serve-auto')
                self.assertEqual(serve[-2], f'0:{5*GIB},1:{6*GIB}')
                self.assertEqual(serve[-1], f'{22*GIB}:{20*GIB}:200:100')
                adapter.offload_to_ram()
                reservations = resources.snapshot()['reservations']
                self.assertNotIn(adapter.owner+':device', reservations)
                self.assertIn(adapter.owner+':context', reservations)
                adapter.close()
                self.assertEqual(resources.snapshot()['reservations'], {})
