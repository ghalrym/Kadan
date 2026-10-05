"""H3 native contract and resource ownership without model weights."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

from api.inference.h3 import H3Provider, H3_REVISION, GIB, sampling_arguments
from api.inference.resources import ResourceCancelled, ResourceExhausted, ResourceManager


def spec(**changes):
    values = dict(prompt='A waterfall', negative_prompt='', duration=8, fps=24,
                  resolution='768p', aspect='16:9', seed=42)
    return SimpleNamespace(**(values | changes))


class H3Tests(unittest.TestCase):
    def test_rejects_unsupported_settings(self):
        for changes in [dict(fps=30), dict(duration=3), dict(duration=16),
                        dict(resolution='1080p'), dict(negative_prompt='blur')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                H3Provider().validate(spec(**changes))

    def test_official_sampler_contract(self):
        args = sampling_arguments(spec(), Path('/tmp/video.mp4'))
        self.assertEqual(args['task'], 't2va')
        self.assertEqual(args['conditions'], [])
        self.assertEqual(args['num_inference_steps'], 50)
        self.assertEqual((args['flow_shift'], args['audio_flow_shift']), (12.0, 3.0))
        self.assertEqual(args['target'], dict(short_edge=768, aspect_ratio='16:9', duration_seconds=8))
        self.assertNotIn('negative_prompt', args)

    def test_ref2va_is_not_misrepresented_as_text_only(self):
        with self.assertRaisesRegex(ValueError, 'reference inputs'):
            H3Provider('h3-ref2va').validate(spec())

    def test_memory_is_leased_until_worker_cleanup(self):
        for failure in (None, RuntimeError('worker failed'), ResourceCancelled('cancel')):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                resources = ResourceManager(400 * GIB, {0: 24 * GIB})
                entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=144_000_000_000)
                observed = []

                def run(*args):
                    state = resources.snapshot()
                    observed.append(state)
                    self.assertEqual(state['exclusive_owner'], 'video:h3')
                    self.assertEqual(state['reservations']['video:h3']['active_leases'], 1)
                    if failure:
                        raise failure

                with patch('api.inference.h3.model_manager.get_checkpoint', create=True, return_value=(entry, Path(directory))), \
                     patch('api.inference.h3.runtime.ensure_resources', return_value=resources), \
                     patch.object(H3Provider, '_run', side_effect=run):
                    if failure:
                        with self.assertRaises(type(failure)):
                            H3Provider().generate(spec(), Path(directory) / 'out.mp4', threading.Event())
                    else:
                        H3Provider().generate(spec(), Path(directory) / 'out.mp4', threading.Event())
                self.assertTrue(observed)
                self.assertEqual(resources.snapshot()['reservations'], {})
                self.assertIsNone(resources.snapshot()['exclusive_owner'])

    def test_insufficient_host_memory_does_not_start_worker(self):
        resources = ResourceManager(32 * GIB, {0: 24 * GIB})
        entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=144_000_000_000)
        with patch('api.inference.h3.model_manager.get_checkpoint', create=True, return_value=(entry, Path('/tmp'))), \
             patch('api.inference.h3.runtime.ensure_resources', return_value=resources), \
             patch.object(H3Provider, '_run') as run:
            with self.assertRaises(ResourceExhausted):
                H3Provider().generate(spec(), Path('/tmp/out.mp4'), threading.Event())
            run.assert_not_called()
        self.assertIsNone(resources.snapshot()['exclusive_owner'])

    def test_worker_is_offline_and_reaped_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.mp4'
            process = SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0, wait=lambda **kwargs: 0)

            def launch(command, **kwargs):
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                self.assertEqual(kwargs['env']['TRANSFORMERS_OFFLINE'], '1')
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(command[1:3], ['-m', 'api.inference.h3_worker'])
                output.write_bytes(b'fixture-output')
                return process

            with patch('api.inference.h3.subprocess.Popen', side_effect=launch), \
                 patch('api.inference.h3.stop_process_group') as cleanup:
                H3Provider()._run(spec(), Path(directory), output, threading.Event(), 0)
                cleanup.assert_called_once_with(process)

    def test_cancellation_removes_partial_and_reaps_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.mp4'
            cancelled = threading.Event()
            process = SimpleNamespace(pid=123456, returncode=None, poll=lambda: None)

            def launch(*args, **kwargs):
                output.write_bytes(b'partial')
                cancelled.set()
                return process

            with patch('api.inference.h3.subprocess.Popen', side_effect=launch), \
                 patch('api.inference.h3.stop_process_group') as cleanup:
                with self.assertRaises(ResourceCancelled):
                    H3Provider()._run(spec(), Path(directory), output, cancelled, 0)
                cleanup.assert_called_once_with(process)
                self.assertFalse(output.exists())
