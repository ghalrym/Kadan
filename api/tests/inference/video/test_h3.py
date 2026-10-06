"""H3 native contract and resource ownership without model weights."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import weakref
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch
from api.inference.video.h3_pipeline import H3Session

from api.inference.video.h3 import H3Provider, H3_REVISION, GIB, sampling_arguments
from api.inference.video import VideoSpec
from api.services.video_jobs import VideoJobs
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted, ResourceManager


def spec(**changes):
    values = dict(prompt='A waterfall', negative_prompt='', duration=8, fps=24,
                  resolution='768p', aspect='16:9', seed=42)
    return SimpleNamespace(**(values | changes))


class H3Tests(unittest.TestCase):
    def setUp(self):
        # These lifecycle fixtures do not encode media. The container build
        # executes both real binaries; the missing-tool test overrides this.
        tools = patch('api.inference.video.h3.shutil.which', side_effect=lambda name: f'/usr/bin/{name}')
        tools.start()
        self.addCleanup(tools.stop)

    def test_rejects_unsupported_settings(self):
        for changes in [dict(fps=30), dict(duration=3), dict(duration=16),
                        dict(resolution='1080p'), dict(negative_prompt='blur')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                H3Provider().validate(spec(**changes))

    def test_official_sampler_contract(self):
        args = sampling_arguments(spec(), Path('/tmp/video.mp4'))
        self.assertEqual(args['task'], 't2va')
        self.assertEqual(args['conditions'], [])
        self.assertEqual(args['num_inference_steps'], 5)
        self.assertEqual((args['flow_shift'], args['audio_flow_shift']), (12.0, 3.0))
        self.assertEqual(args['target'], dict(short_edge=768, aspect_ratio='16:9', duration_seconds=8))
        self.assertNotIn('negative_prompt', args)

    def test_ref2va_is_not_misrepresented_as_text_only(self):
        with self.assertRaisesRegex(ValueError, 'reference inputs'):
            H3Provider('h3-ref2va').validate(spec())

    def test_memory_is_leased_until_worker_cleanup(self):
        for failure in (None, RuntimeError('worker failed'), ResourceCancelled('cancel')):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                resources = ResourceManager(400 * GIB, {0: 18 * GIB})
                entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=144_000_000_000)
                observed = []

                def run(*args):
                    state = resources.snapshot()
                    observed.append(state)
                    self.assertIsNone(state['exclusive_owner'])
                    for name in ('host', 'context', 'device'):
                        self.assertEqual(state['reservations']['video:h3:' + name]['active_leases'], 1)
                    self.assertEqual(state['reservations']['video:h3:device']['device_bytes'], {0: 17 * GIB})
                    self.assertEqual(state['reservations']['video:h3:context']['device_bytes'], {0: GIB})
                    if failure:
                        raise failure

                with patch('api.inference.video.h3.model_manager.get_checkpoint', create=True, return_value=(entry, Path(directory))), \
                     patch('api.inference.video.h3.runtime.ensure_resources', return_value=resources), \
                     patch.object(H3Provider, '_run', side_effect=run):
                    if failure:
                        with self.assertRaises(type(failure)):
                            H3Provider().generate(spec(), Path(directory) / 'out.mp4', threading.Event())
                    else:
                        provider = H3Provider()
                        provider.generate(spec(), Path(directory) / 'out.mp4', threading.Event())
                        self.assertIn('video:h3:host', resources.snapshot()['reservations'])
                        provider.close()
                self.assertTrue(observed)
                self.assertEqual(resources.snapshot()['reservations'], {})
                self.assertIsNone(resources.snapshot()['exclusive_owner'])

    def test_insufficient_host_memory_does_not_start_worker(self):
        resources = ResourceManager(32 * GIB, {0: 200 * GIB})
        entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=144_000_000_000)
        with patch('api.inference.video.h3.model_manager.get_checkpoint', create=True, return_value=(entry, Path('/tmp'))), \
             patch('api.inference.video.h3.runtime.ensure_resources', return_value=resources), \
             patch.object(H3Provider, '_run') as run:
            with self.assertRaises(ResourceExhausted):
                H3Provider().generate(spec(), Path('/tmp/out.mp4'), threading.Event())
            run.assert_not_called()
        self.assertIsNone(resources.snapshot()['exclusive_owner'])

    def test_direct_renderer_stays_in_process_and_publishes_atomically(self):
        from api.inference.video import h3_pipeline
        import os
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / '.job.partial.mp4'
            caller_pid = os.getpid()
            def render(checkpoint, sampling, device, cancel, session=None):
                self.assertEqual(os.getpid(), caller_pid)
                self.assertEqual(device, 0)
                target = Path(sampling['output_path']) / sampling['output_file_name']
                self.assertFalse(output.exists())
                target.write_bytes(b'validated-test-output')
            with patch.object(h3_pipeline, 'render', side_effect=render), \
                 patch('subprocess.Popen', side_effect=AssertionError('No model process')):
                H3Provider()._run(spec(), Path(directory), output, threading.Event(), [0])
            self.assertEqual(output.read_bytes(), b'validated-test-output')
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_cooperative_cancellation_cleans_private_output(self):
        from api.inference.video import h3_pipeline
        with tempfile.TemporaryDirectory() as directory:
            event = threading.Event()
            output = Path(directory) / 'out.mp4'
            def render(checkpoint, sampling, device, cancel, session=None):
                (Path(sampling['output_path']) / sampling['output_file_name']).write_bytes(b'partial')
                cancel.set()
            with patch.object(h3_pipeline, 'render', side_effect=render):
                with self.assertRaises(ResourceCancelled):
                    H3Provider()._run(spec(), Path(directory), output, event, [0])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_turbo_must_remain_dynamic(self):
        from api.inference.video.h3_pipeline import require_turbo
        pipeline = MagicMock()
        for active in ({}, {'transformer': []}, {'transformer': [{'merged': True, 'strengths': [1.0]}]}):
            pipeline.get_lora_status.return_value = {'active': active}
            with self.assertRaises(RuntimeError):
                require_turbo(pipeline)
        pipeline.get_lora_status.return_value = {'active': {'transformer': [{'merged': False, 'strengths': [1.0]}]}}
        require_turbo(pipeline)

    def test_renderer_failure_cleans_staging_before_return(self):
        from api.inference.video import h3_pipeline
        with tempfile.TemporaryDirectory() as directory:
            def fail(checkpoint, sampling, device, cancel, session=None):
                (Path(sampling['output_path']) / sampling['output_file_name']).write_bytes(b'partial')
                raise RuntimeError('native failure')
            with patch.object(h3_pipeline, 'render', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'native failure'):
                    H3Provider()._run(spec(), Path(directory), Path(directory) / 'out.mp4', threading.Event(), [0])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_precancelled_request_never_enters_native_pipeline(self):
        from api.inference.video import h3_pipeline
        event = threading.Event()
        event.set()
        with tempfile.TemporaryDirectory() as directory, patch.object(h3_pipeline, 'render') as render:
            with self.assertRaises(ResourceCancelled):
                H3Provider()._run(spec(), Path(directory), Path(directory) / 'out.mp4', event, [0])
            render.assert_not_called()

    def test_native_frames_release_before_cuda_cache_and_lease_return(self):
        from api.inference.video import h3_pipeline

        class TensorOwner:
            pass

        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                retained = []
                def native(*args):
                    owner = TensorOwner()
                    owner.cycle = owner
                    retained.append(weakref.ref(owner))
                    if cancelled:
                        raise ResourceCancelled('cancel inside native forward')

                def empty_cache():
                    self.assertIsNone(retained[0]())

                with patch.object(h3_pipeline, '_render', side_effect=native), \
                     patch('torch.cuda.device'), \
                     patch('torch.cuda.empty_cache', side_effect=empty_cache), \
                     patch('torch.accelerator.memory.empty_host_cache', side_effect=empty_cache) as host_cache, \
                     patch.object(h3_pipeline, 'trim_cpu_heap', side_effect=empty_cache) as trim:
                    if cancelled:
                        with self.assertRaises(ResourceCancelled):
                            h3_pipeline.render(None, None, 0, threading.Event())
                    else:
                        h3_pipeline.render(None, None, 0, threading.Event())
                host_cache.assert_called_once_with()
                trim.assert_called_once_with()

    def test_heap_trim_is_optional_and_uses_process_allocator(self):
        from api.inference.video import h3_pipeline
        library = SimpleNamespace(malloc_trim=MagicMock())
        with patch.object(h3_pipeline.sys, 'platform', 'linux'), \
             patch.object(h3_pipeline.ctypes, 'CDLL', return_value=library) as load:
            h3_pipeline.trim_cpu_heap()
            load.assert_called_once_with(None)
            library.malloc_trim.assert_called_once_with(0)
        with patch.object(h3_pipeline.sys, 'platform', 'linux'), \
             patch.object(h3_pipeline.ctypes, 'CDLL', return_value=SimpleNamespace()):
            h3_pipeline.trim_cpu_heap()
        with patch.object(h3_pipeline.sys, 'platform', 'darwin'), \
             patch.object(h3_pipeline.ctypes, 'CDLL') as load:
            h3_pipeline.trim_cpu_heap()
            load.assert_not_called()

    def test_retained_session_parks_and_reuses_same_host_object_then_pressure_evicts(self):
        resources = ResourceManager(400 * GIB, {0: 18 * GIB})
        entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=69_000_000_000)
        session = MagicMock()
        provider = H3Provider()
        def run(*args):
            provider._session = session
        with patch('api.inference.video.h3.model_manager.get_checkpoint', return_value=(entry, Path('/fixture'))), patch(
                'api.inference.video.h3.runtime.ensure_resources', return_value=resources), patch.object(provider, '_run', side_effect=run):
            provider.generate(spec(), Path('/unused'), threading.Event())
            host = provider._host
            resources.offload_inactive_devices('speech')
            session.park.assert_called_once_with(GIB)
            session.close.assert_not_called()
            self.assertEqual(set(resources.snapshot()['reservations']), {'video:h3:host', 'video:h3:context'})
            provider.generate(spec(), Path('/unused'), threading.Event())
            self.assertIs(provider._host, host)
            self.assertIs(provider._session, session)
            resources.offload_inactive_devices('llm')
            pressure = resources.reserve('large', 'llm', host_bytes=400 * GIB)
            session.close.assert_called_once()
            self.assertEqual(set(resources.snapshot()['reservations']), {'large'})
            pressure.release()

    def test_active_h3_reservations_cannot_be_parked_or_pressure_evicted(self):
        resources = ResourceManager(400 * GIB, {0: 18 * GIB})
        entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=69_000_000_000)
        provider = H3Provider()
        def run(*args):
            with self.assertRaises(ResourceBusy):
                resources.offload_inactive_devices('speech')
            with self.assertRaises(ResourceExhausted):
                resources.reserve('large', 'llm', host_bytes=400 * GIB)
        with patch('api.inference.video.h3.model_manager.get_checkpoint', return_value=(entry, Path('/fixture'))), patch(
                'api.inference.video.h3.runtime.ensure_resources', return_value=resources), patch.object(provider, '_run', side_effect=run):
            provider.generate(spec(), Path('/unused'), threading.Event())
            provider.close()

    def test_parking_moves_unregistered_turbo_cache_to_host(self):
        session = H3Session('/fixture', 0)
        component = torch.nn.Linear(2, 2)
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device.type = 'cuda'
        tensor.to.return_value = torch.ones(2)
        session.pipeline = SimpleNamespace(modules={'transformer': component}, lora_adapters={'turbo': {'a': tensor}})
        with patch('torch.cuda.device'), patch('torch.cuda.synchronize'), patch('torch.cuda.empty_cache'), patch(
                'torch.cuda.memory_allocated', return_value=128 * 1024**2):
            session.park(GIB)
        tensor.to.assert_called_once_with('cpu')
        self.assertIs(session.pipeline.lora_adapters['turbo']['a'], tensor.to.return_value)
        self.assertTrue(session.parked)
