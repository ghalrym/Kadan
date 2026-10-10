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
from api.inference.video.h3_worker import H3Provider, GIB, PROCESS_HOST_BUDGET, validate_artifact, decode_response, h3_gpu_budgets
from api.inference.line_protocol import LineProtocolError


class Worker:
    def __init__(self):
        self.started = self.stopped = False
        self.fail_stop = False
        self.cancel = False
        self.bad_ledger = False
        self.cache_budget = 0
        self.gpu_budget = 3*GIB
        self.controls = []
        self.bad_park = False
        self.bad_resume = False
        self.cancel_resume = False
        self.calls = []
    def start(self, command, *, env=None):
        assert env['KADAN_H3_DEVICES'] == '0,1'
        self.cache_budget = int(env['KADAN_H3_WEIGHT_CACHE_BYTES'])
        self.gpu_budget = int(env['KADAN_H3_GPU_BUDGET_BYTES'])
        self.gpu_budgets = env['KADAN_H3_GPU_BUDGETS']
        self.started = True
    def stop(self):
        if self.fail_stop:
            raise TimeoutError('unconfirmed child')
        self.stopped = True
    def exchange(self, request, timeout, cancel=None):
        if 'control' in request:
            self.controls.append(request['control'])
            if request['control']=='resume' and self.cancel_resume:
                cancel.set()
                raise ResourceCancelled('resume cancelled')
            return dict(state='parked' if request['control']=='park' else 'resumed',
                resident_bytes=self.cache_budget, device_resident_bytes=[1,0] if
                (self.bad_park and request['control']=='park') or (self.bad_resume and request['control']=='resume') else [0,0])
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
        weights = self.gpu_budget - 3*GIB
        metadata = 8*1024**2 if weights else 0
        return dict(output=str(raw), width=864, height=480, frames=107,
                    audio=True, resident_bytes=self.cache_budget+metadata, weight_cache_bytes=self.cache_budget,
                    device_weight_metadata_bytes=metadata,
                    device_resident_bytes=[weights+1, 0] if self.bad_ledger else [weights, weights])


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
            assert sum(item["host_bytes"] for item in state.values()) == PROCESS_HOST_BUDGET + provider._cache_budget
        assert released == ['text'] and len(worker.calls) == 2
        assert worker.calls[0]['short_edge'] == 480 and worker.calls[0]['duration'] == 4
        assert worker.calls[0]['updates'] == 4 and not worker.stopped
        resources.offload_inactive_devices('llm')
        assert not worker.stopped and provider._context is None and provider._host is not None
        self.assertEqual(worker.controls, ['park'])
        self.assertEqual(sum(sum(r['device_bytes'].values()) for r in resources.snapshot()['reservations'].values()), 0)
        provider.generate(spec(), self.root/'resumed.mp4', threading.Event())
        self.assertEqual(worker.controls, ['park', 'resume'])
        self.assertFalse(worker.stopped)
        provider.close()
        assert not list(self.root.glob('.h3-*'))

    def test_cache_capacity_is_frozen_and_admitted_before_spawn(self):
        for extra, expected in ((0, 0), (3*GIB, 3*GIB), (64*GIB, 32*GIB)):
            with self.subTest(extra=extra):
                _, worker, provider = setup(self)
                provider.resources = ResourceManager(PROCESS_HOST_BUDGET+extra, {0:8*GIB, 1:8*GIB})
                provider.load(threading.Event())
                self.assertEqual(worker.cache_budget, expected)
                self.assertEqual(provider._cache_budget, expected)
                self.assertEqual(sum(r['host_bytes'] for r in provider.resources.snapshot()['reservations'].values()), PROCESS_HOST_BUDGET+expected)
                provider.close()

    def test_cache_report_cannot_hide_unowned_or_execution_memory(self):
        response = dict(output='/unused', width=864, height=480, frames=107, audio=True,
            resident_bytes=2*GIB, weight_cache_bytes=2*GIB, device_resident_bytes=[0,0], device_weight_metadata_bytes=0)
        with self.assertRaisesRegex(LineProtocolError, 'accounting'):
            validate_artifact(response, Path('/unused'), spec(), GIB)
        response['weight_cache_bytes'] = GIB
        with self.assertRaisesRegex(LineProtocolError, 'accounting'):
            validate_artifact(response, Path('/unused'), spec(), GIB)

    def test_cancelled_cache_admission_never_spawns(self):
        resources, worker, provider = setup(self)
        cancel = threading.Event()
        with patch.object(resources, 'reserve', side_effect=ResourceCancelled('cancelled')) as reserve:
            with self.assertRaises(ResourceCancelled):
                provider.load(cancel)
            self.assertIs(reserve.call_args.kwargs['cancel_event'], cancel)
            self.assertEqual(reserve.call_args.kwargs['host_bytes'], PROCESS_HOST_BUDGET+32*GIB)
        self.assertFalse(worker.started)
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_bad_park_ack_reaps_before_host_or_context_release(self):
        resources, worker, provider = setup(self)
        provider.load(threading.Event())
        worker.bad_park = worker.fail_stop = True
        with self.assertRaises(TimeoutError):
            provider.offload_to_ram()
        self.assertIsNotNone(provider._host)
        self.assertIsNotNone(provider._context)
        self.assertTrue(resources.snapshot()['reservations'])
        with self.assertRaises(InferenceFailure):
            provider.check_execution_state()
        worker.fail_stop = False
        provider.close()
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_control_failure_with_confirmed_reap_releases_all_ownership(self):
        for control in ('park', 'resume', 'cancel_resume'):
            with self.subTest(control=control):
                resources, worker, provider = setup(self)
                provider.load(threading.Event())
                if control=='park':
                    worker.bad_park=True
                    action=provider.offload_to_ram
                else:
                    provider.offload_to_ram()
                    worker.bad_resume=control=='resume'
                    worker.cancel_resume=control=='cancel_resume'
                    action=lambda:provider.load(threading.Event())
                with self.assertRaises((LineProtocolError,ResourceCancelled)):
                    action()
                self.assertTrue(worker.stopped)
                self.assertEqual(resources.snapshot()['reservations'], {})

    def test_gpu_budget_uses_configured_capacity_and_honors_explicit_limit(self):
        resources=ResourceManager(200*GIB,{0:22*GIB,1:20*GIB})
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(h3_gpu_budgets(resources,[0,1]),{0:22*GIB,1:20*GIB})
            self.assertEqual(h3_gpu_budgets(resources,[0]),{0:22*GIB})
            self.assertEqual(h3_gpu_budgets(ResourceManager(200*GIB,{0:48*GIB}),[0]),{0:24*GIB})
        with patch.dict(os.environ, {'KADAN_H3_GPU_BUDGET_BYTES':str(3*GIB)}):
            self.assertEqual(h3_gpu_budgets(resources,[0,1]),{0:3*GIB,1:3*GIB})
        for value in ('0','-1','x',str(2*GIB),str(23*GIB)):
            with self.subTest(value=value),patch.dict(os.environ, {'KADAN_H3_GPU_BUDGET_BYTES':value}),self.assertRaises(InferenceFailure):
                h3_gpu_budgets(resources,[0,1])

    def test_unequal_device_handoff_keeps_host_and_restores_exact_budgets(self):
        _, worker, provider = setup(self)
        resources = ResourceManager(200*GIB, {0:22*GIB, 1:12*GIB})
        provider.resources = resources
        with patch.dict(os.environ, {}, clear=True):
            provider.load(threading.Event())
            self.assertEqual(worker.gpu_budgets, f'0:{22*GIB},1:{12*GIB}')
            def devices():
                return [v['device_bytes'] for v in resources.snapshot()['reservations'].values() if v['device_bytes']]
            self.assertEqual(devices(), [{0:20*GIB, 1:10*GIB}])
            provider.offload_to_ram()
            self.assertEqual(devices(), [])
            self.assertIsNotNone(provider._host)
            provider.load(threading.Event())
            self.assertEqual(devices(), [{0:20*GIB, 1:10*GIB}])
            self.assertEqual(worker.controls, ['park','resume'])
            provider.close()
            self.assertEqual(resources.snapshot()['reservations'], {})

    def test_single_gpu_retention_and_metadata_are_bound_to_admission(self):
        response=dict(output='/unused',width=864,height=480,frames=107,audio=True,
            weight_cache_bytes=0,resident_bytes=4*1024**2,device_weight_metadata_bytes=4*1024**2,
            device_resident_bytes=[GIB,0])
        with self.assertRaisesRegex(LineProtocolError,'device memory'):
            validate_artifact(response,Path('/unused'),spec(),device_budgets={1:GIB})
        response['device_resident_bytes']=[0,GIB]
        response['device_weight_metadata_bytes']=0
        with self.assertRaisesRegex(LineProtocolError,'accounting'):
            validate_artifact(response,Path('/unused'),spec(),device_budgets={1:GIB})

    def test_cancelled_resume_admission_reaps_retained_host_cache(self):
        resources,worker,provider=setup(self)
        provider.load(threading.Event())
        provider.offload_to_ram()
        self.assertIsNotNone(provider._host)
        with patch.object(resources,'reserve',side_effect=ResourceCancelled('waiting cancelled')):
            with self.assertRaises(ResourceCancelled):
                provider.load(threading.Event())
        self.assertTrue(worker.stopped)
        self.assertEqual(resources.snapshot()['reservations'],{})


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
