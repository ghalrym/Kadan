"""Admission, FIFO handoff and child ownership without model or GPU execution."""
import sys
from api.services.video_jobs import VideoJobs
from api.inference.video.video_requests import VideoRequests
from api.inference.video.h3_worker import H3Process
from pathlib import Path
import threading
import shutil
import subprocess
import json
import wave

import os
import tempfile
import unittest
from unittest.mock import patch

from api.inference.errors import InferenceFailure
from api.inference.resources import ResourceManager, ResourceCancelled, ResourceExhausted
from api.inference.video import VideoSpec
from api.inference.video.h3_worker import H3Provider, GIB, PROCESS_HOST_BUDGET, validate_artifact, decode_response
from api.inference.line_protocol import LineProtocolError


class Worker:
    def __init__(self):
        self.started = self.stopped = False
        self.fail_stop = False
        self.cancel = False
        self.bad_ledger = False
        self.calls = []
    def start(self, command, *, env=None):
        assert env['KADAN_H3_DEVICES'] == '0,1'
        self.started = True
    def stop(self):
        if self.fail_stop:
            raise TimeoutError('unconfirmed child')
        self.stopped = True
    def exchange(self, request, timeout, cancel=None):
        self.calls.append(request)
        if self.cancel:
            cancel.set()
            raise ResourceCancelled('cancelled')
        raw = Path(request['output'])
        header = b'YUV4MPEG2 W864 H480 F24:1 Ip A1:1 C444 XCOLORRANGE=FULL\n'
        with raw.open('wb') as stream:
            stream.write(header)
            stream.truncate(len(header) + 107*(6 + 864*480*3))
        with wave.open(str(raw)+'.wav','wb') as audio:
            audio.setparams((2,2,32000,0,'NONE','NONE'));audio.writeframes(b'\0'*round(107*5/3)*800*4)
        return dict(output=str(raw), width=864, height=480, frames=107,
                    audio=True, resident_bytes=0, device_resident_bytes=[1, 0] if self.bad_ledger else [0, 0])


def setup(monkeypatch, **kwargs):
    monkeypatch.setenv('KADAN_H3_DEVICES', '0,1')
    resources = ResourceManager(200*GIB, {0: 8*GIB, 1: 8*GIB})
    worker = Worker()
    provider = H3Provider(resources=resources, resolve=lambda _: ['fake-worker'],
                          process_factory=lambda: worker, **kwargs)
    def encode(raw, target, frames, cancel):
        assert provider._execution is not None and provider._context is not None
        assert frames == 107
        target.write_bytes(b'codec-result')
    monkeypatch.setattr(provider, '_encode', encode)
    return resources, worker, provider


def spec():
    return VideoSpec('red ball', duration=4, resolution='480p')


class H3WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
    def setenv(self, name, value):
        context = patch.dict(os.environ, {name: value})
        context.start(); self.addCleanup(context.stop)
    def setattr(self, obj, name, value):
        context = patch.object(obj, name, value)
        context.start(); self.addCleanup(context.stop)

    def test_native_request_retention_and_handoff(self):
        resources, worker, provider = setup(self)
        released = []
        other = resources.reserve('text', 'llm', device_bytes={1:GIB}, evict=lambda: released.append('text'))
        for i in range(2):
            output = self.root / f'{i}.mp4'
            provider.generate(spec(), output, threading.Event())
            assert output.read_bytes() == b'codec-result'
            assert provider._execution is None and provider._context is not None
            state = resources.snapshot()["reservations"]
            assert sum(item["host_bytes"] for item in state.values()) == PROCESS_HOST_BUDGET
        assert released == ['text'] and len(worker.calls) == 2
        assert worker.calls[0]['short_edge'] == 480 and worker.calls[0]['duration'] == 4
        assert worker.calls[0]['updates'] == 4 and not worker.stopped
        resources.offload_inactive_devices('llm')
        assert worker.stopped and provider._context is None
        assert not list(self.root.glob('.h3-*'))


    def test_admission_before_spawn(self):
        _, worker, provider = setup(self)
        provider.resources = ResourceManager(2*GIB, {0:8*GIB, 1:8*GIB})
        with self.assertRaises(ResourceExhausted):
            provider.generate(spec(), self.root/'out.mp4', threading.Event())
        assert not worker.started and provider._context is None and provider._execution is None


    def test_cancel_reaps_then_releases(self):
        _, worker, provider = setup(self)
        worker.cancel = True
        with self.assertRaises(ResourceCancelled):
            provider.generate(spec(), self.root/'out.mp4', threading.Event())
        assert worker.stopped and provider._context is None and provider._execution is None
        assert not list(self.root.iterdir())


    def test_failed_reap_retains_ownership_and_blocks_fifo(self):
        _, worker, provider = setup(self)
        worker.cancel = worker.fail_stop = True
        with self.assertRaises(TimeoutError):
            provider.generate(spec(), self.root/'out.mp4', threading.Event())
        assert provider._context is not None and provider._execution is not None
        assert provider._workspace is not None
        with self.assertRaises(InferenceFailure):
            provider.check_execution_state()
        worker.fail_stop = False
        provider.close()
        provider.check_execution_state()
        assert provider._context is None and provider._execution is None
        assert not list(self.root.iterdir())


    def test_nonzero_native_ledger_rejected(self):
        _, worker, provider = setup(self)
        worker.bad_ledger = True
        with self.assertRaises(LineProtocolError):
            provider.generate(spec(), self.root/'out.mp4', threading.Event())
        assert worker.stopped and provider._execution is None
        assert not list(self.root.iterdir())


    def test_existing_output_preserved(self):
        _, worker, provider = setup(self)
        output = self.root/'out.mp4'; output.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            provider.generate(spec(), output, threading.Event())
        assert output.read_bytes() == b'existing' and worker.stopped


    def test_duplicate_and_nonfinite_response_rejected(self):
        for frame in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(LineProtocolError):
                decode_response(frame)

    def test_service_selects_cpp_and_forwards_quarantine(self):
        jobs = VideoJobs(root=self.root)
        provider = jobs.provider('h3-fl2va-int8-turbo')
        self.assertIsInstance(provider, H3Provider)
        provider._quarantined = True
        with self.assertRaises(InferenceFailure):
            VideoRequests(jobs).check_execution_state()

    def test_preflight_rejection_does_not_park_text(self):
        resources, worker, provider = setup(self)
        parked = []
        resources.reserve('text', 'llm', device_bytes={1:GIB}, evict=lambda: parked.append(True))
        def unavailable(_):
            raise InferenceFailure('CUDA worker unavailable')
        provider.resolve = unavailable
        with self.assertRaises(InferenceFailure):
            provider.generate(spec(), self.root/'out.mp4', threading.Event())
        self.assertFalse(parked)
        self.assertFalse(worker.started)


    def test_precancelled_codec_never_spawns(self):
        _, _, provider = setup(self)
        event = threading.Event(); event.set()
        with self.assertRaises(ResourceCancelled):
            H3Provider._encode(provider, self.root/'missing.y4m', self.root/'out.mp4', 107, event)
        self.assertIsNone(provider._codec)


    def test_real_json_transport_handles_unicode_and_stderr(self):
        child = H3Process()
        code = 'import sys,json; r=json.loads(sys.stdin.readline()); sys.stderr.write("x"*20000); print(json.dumps({"length":len(r["prompt"])}),flush=True)'
        try:
            child.start([sys.executable, '-c', code], env=dict(os.environ))
            result = child.exchange({'prompt':'🎬' * 8000}, 5, threading.Event())
            self.assertEqual(result, {'length':8000})
            self.assertLessEqual(len(child.diagnostics), 8192)
        finally:
            child.stop()
        self.assertTrue(child.closed)

    def test_truncated_raw_video_is_rejected(self):
        _, worker, _ = setup(self)
        raw = self.root/'raw.y4m'
        response = worker.exchange({'output':str(raw)}, 1)
        with raw.open('r+b') as stream:
            stream.truncate(raw.stat().st_size - 1)
        with self.assertRaises(LineProtocolError):
            validate_artifact(response, raw, spec())
