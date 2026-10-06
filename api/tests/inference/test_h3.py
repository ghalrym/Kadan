"""H3 native contract and resource ownership without model weights."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from api.inference.h3 import H3Provider, H3_REVISION, GIB, sampling_arguments
from api.inference.video import VideoSpec
from api.services.video_jobs import VideoJobs
from api.inference.resources import ResourceCancelled, ResourceExhausted, ResourceManager


def spec(**changes):
    values = dict(prompt='A waterfall', negative_prompt='', duration=8, fps=24,
                  resolution='768p', aspect='16:9', seed=42)
    return SimpleNamespace(**(values | changes))


class H3Tests(unittest.TestCase):
    def setUp(self):
        # These lifecycle fixtures do not encode media. The container build
        # executes both real binaries; the missing-tool test overrides this.
        tools = patch('api.inference.h3.shutil.which', side_effect=lambda name: f'/usr/bin/{name}')
        tools.start()
        self.addCleanup(tools.stop)
        interpreter = patch.dict('os.environ', {'KADAN_H3_PYTHON': sys.executable})
        interpreter.start()
        self.addCleanup(interpreter.stop)
        config = SimpleNamespace(MiniMaxH3PipelineConfig=lambda: SimpleNamespace(
            dit_config=SimpleNamespace(arch_config=SimpleNamespace())))
        native_config = patch.dict(sys.modules, {
            'sglang.multimodal_gen.configs.pipeline_configs.minimax_h3': config})
        native_config.start()
        self.addCleanup(native_config.stop)

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
                resources = ResourceManager(400 * GIB, {0: 18 * GIB, 1: 17 * GIB})
                entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=144_000_000_000)
                observed = []

                def run(*args):
                    state = resources.snapshot()
                    observed.append(state)
                    self.assertEqual(state['exclusive_owner'], 'video:h3')
                    self.assertEqual(state['reservations']['video:h3']['active_leases'], 1)
                    self.assertEqual(state['reservations']['video:h3']['device_bytes'], {0: 18 * GIB, 1: 17 * GIB})
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
        resources = ResourceManager(32 * GIB, {0: 200 * GIB})
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
            output = Path(directory) / '.job.partial.mp4'
            process = SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0, wait=lambda **kwargs: 0)
            staging = []

            def launch(command, **kwargs):
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                self.assertEqual(kwargs['env']['TRANSFORMERS_OFFLINE'], '1')
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(command[1:3], ['-m', 'api.inference.h3_worker'])
                payload = json.loads(Path(command[-1]).read_text())
                args = payload['sampling']
                self.assertEqual(args['output_file_name'], 'video.mp4')
                rendered = Path(args['output_path']) / args['output_file_name']
                self.assertNotEqual(rendered, output)
                self.assertEqual(rendered.parent.parent, output.parent)
                rendered.write_bytes(b'fixture-output')
                (rendered.parent / 'sidecar.wav').write_bytes(b'audio')
                staging.append(rendered.parent)
                return process

            with patch('api.inference.h3.subprocess.Popen', side_effect=launch), \
                 patch('api.inference.h3.stop_process_group') as cleanup:
                H3Provider()._run(spec(), Path(directory), output, threading.Event(), [0])
                cleanup.assert_called_once_with(process)
            self.assertEqual(output.read_bytes(), b'fixture-output')
            self.assertFalse(staging[0].exists())

    def test_cancellation_removes_partial_and_reaps_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.mp4'
            cancelled = threading.Event()
            process = SimpleNamespace(pid=123456, returncode=None, poll=lambda: None)
            staging = []

            def launch(command, **kwargs):
                args = json.loads(Path(command[-1]).read_text())['sampling']
                root = Path(args['output_path'])
                (root / args['output_file_name']).write_bytes(b'partial')
                (root / 'sidecar.wav').write_bytes(b'partial')
                staging.append(root)
                cancelled.set()
                return process

            with patch('api.inference.h3.subprocess.Popen', side_effect=launch), \
                 patch('api.inference.h3.stop_process_group') as cleanup:
                with self.assertRaises(ResourceCancelled):
                    H3Provider()._run(spec(), Path(directory), output, cancelled, [0])
                cleanup.assert_called_once_with(process)
                self.assertFalse(output.exists())
                self.assertFalse(staging[0].exists())

    def test_native_generator_contract_and_shutdown_on_failure(self):
        # Load the optional worker against a tiny native API fixture, never SGLang/weights.
        module_name = 'sglang.multimodal_gen.runtime.entrypoints.diffusion_generator'
        generator = MagicMock()
        generator.list_loras.return_value = {'active': {'transformer': [{'merged': False, 'strengths': [1.0]}]}}
        generator.generate.side_effect = RuntimeError('native failure')
        factory = MagicMock()
        factory.from_pretrained.return_value = generator
        fixture = SimpleNamespace(DiffGenerator=factory)
        location = Path(__file__).resolve().parents[2] / 'inference' / 'h3_worker.py'
        definition = importlib.util.spec_from_file_location('h3_worker_contract', location)
        worker = importlib.util.module_from_spec(definition)
        modules = patch.dict(sys.modules, {module_name: fixture,
             'sglang.multimodal_gen.runtime.entrypoints.utils': SimpleNamespace(GenerationResult=SimpleNamespace)})
        modules.start()
        self.addCleanup(modules.stop)
        definition.loader.exec_module(worker)
        arguments = sampling_arguments(spec(), Path('/tmp/out.mp4'))
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            worker.run(dict(checkpoint='/models/local', num_gpus=2, sampling=arguments))
        kwargs = factory.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['model_path'], '/models/local/FL2VA')
        self.assertTrue(kwargs['local_mode'])
        self.assertEqual(kwargs['backend'], 'sglang')
        self.assertFalse(kwargs['enable_torch_compile'])
        self.assertFalse(kwargs['pipeline_config'].dit_config.arch_config.qkv_checkpoint_grouped)
        generator.generate.assert_called_once_with(sampling_params_kwargs=arguments)
        generator.shutdown.assert_called_once_with()

    def test_turbo_must_match_layers_and_remain_dynamic(self):
        location = Path(__file__).resolve().parents[2] / 'inference' / 'h3_worker.py'
        definition = importlib.util.spec_from_file_location('h3_turbo_contract', location)
        worker = importlib.util.module_from_spec(definition)
        definition.loader.exec_module(worker)
        generator = MagicMock()
        for active in ({}, {'transformer': []},
                       {'transformer': [{'merged': True, 'strengths': [1.0]}]},
                       {'transformer': [{'merged': False, 'strengths': [0.0]}]}):
            generator.list_loras.return_value = {'active': active}
            with self.subTest(active=active), self.assertRaisesRegex(RuntimeError, 'Turbo'):
                worker.require_turbo(generator)
        generator.list_loras.return_value = {'active': {
            'transformer': [{'merged': False, 'strengths': [1.0]}]}}
        worker.require_turbo(generator)

    def test_missing_media_tools_fails_before_memory_admission(self):
        with patch('api.inference.h3.shutil.which', return_value=None), \
             patch('api.inference.h3.runtime.ensure_resources') as resources:
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg, ffprobe'):
                H3Provider().generate(spec(), Path('/tmp/unused.mp4'), threading.Event())
            resources.assert_not_called()

    def test_nonempty_failed_worker_output_and_sidecars_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / '.job.partial.mp4'
            staging = []
            process = SimpleNamespace(pid=123456, returncode=1, poll=lambda: 1)
            def launch(command, **kwargs):
                args = json.loads(Path(command[-1]).read_text())['sampling']
                root = Path(args['output_path'])
                (root / args['output_file_name']).write_bytes(b'invalid-av')
                (root / 'sidecar.wav').write_bytes(b'invalid-audio')
                staging.append(root)
                return process
            with patch('api.inference.h3.subprocess.Popen', side_effect=launch), \
                 patch('api.inference.h3.stop_process_group'):
                with self.assertRaisesRegex(RuntimeError, 'native worker failed'):
                    H3Provider()._run(spec(), Path(directory), output, threading.Event(), [0])
            self.assertFalse(output.exists())
            self.assertFalse(staging[0].exists())

    def test_native_none_result_rejects_nonempty_invalid_video(self):
        generator = MagicMock()
        generator.list_loras.return_value = {'active': {'transformer': [{'merged': False, 'strengths': [1.0]}]}}
        generator.generate.return_value = None
        factory = MagicMock()
        factory.from_pretrained.return_value = generator
        location = Path(__file__).resolve().parents[2] / 'inference' / 'h3_worker.py'
        definition = importlib.util.spec_from_file_location('h3_none_result', location)
        worker = importlib.util.module_from_spec(definition)
        modules = patch.dict(sys.modules, {
            'sglang.multimodal_gen.runtime.entrypoints.diffusion_generator': SimpleNamespace(DiffGenerator=factory),
            'sglang.multimodal_gen.runtime.entrypoints.utils': SimpleNamespace(GenerationResult=SimpleNamespace),
        })
        modules.start()
        self.addCleanup(modules.stop)
        definition.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'video.mp4'
            output.write_bytes(b'invalid-but-nonempty')
            with self.assertRaisesRegex(RuntimeError, 'audiovisual validation failed'):
                worker.run(dict(checkpoint='/models/local', num_gpus=2, sampling=sampling_arguments(spec(), output)))
            generator.shutdown.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, 'invalid validated output path'):
                worker.accept_result(SimpleNamespace(output_file_path='/outside.mp4'), sampling_arguments(spec(), output))
            validated = output.with_name('validated.mp4')
            validated.write_bytes(b'validated')
            worker.accept_result(SimpleNamespace(output_file_path=str(validated)), sampling_arguments(spec(), output))
            self.assertEqual(output.read_bytes(), b'validated')

    def test_shared_video_queue_executes_h3_and_publishes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = ResourceManager(400 * GIB, {0: 18 * GIB, 1: 17 * GIB})
            entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=144_000_000_000)
            def run(settings, checkpoint, output, cancellation, device):
                output.write_bytes(b'audiovisual-fixture')
            jobs = VideoJobs(root=root)
            with patch('api.inference.h3.model_manager.get_checkpoint', return_value=(entry, root)), \
                 patch('api.inference.h3.runtime.ensure_resources', return_value=resources), \
                 patch.object(H3Provider, '_run', side_effect=run):
                submitted = jobs.submit('h3-fl2va-int8-turbo', VideoSpec(prompt='A river', resolution='768p'))
                jobs._thread.join(timeout=5)
            completed = jobs.get(submitted.id)
            self.assertEqual(completed.status, 'Done')
            self.assertEqual((root / f'{submitted.id}.mp4').read_bytes(), b'audiovisual-fixture')
            self.assertEqual(resources.snapshot()['reservations'], {})
