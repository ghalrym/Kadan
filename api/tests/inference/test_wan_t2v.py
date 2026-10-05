"""Original T2V-A14B native calls and API dispatch without weights."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from api.inference.video import VideoSpec
from api.inference.wan import WanProvider
from api.services.video_jobs import VideoJobs


class WanT2VTests(unittest.TestCase):
    def test_dispatch_selects_the_original_text_only_task(self):
        provider = VideoJobs._provider('wan22-t2v-a14b')
        self.assertIsInstance(provider, WanProvider)
        self.assertEqual(provider.task, 't2v-A14B')
        self.assertEqual(provider.model_id, 'wan22-t2v-a14b')

    def test_native_worker_uses_pinned_sampler_and_exact_output_frame_count(self):
        config = SimpleNamespace(sample_shift=12.0, sample_steps=40, sample_guide_scale=(3.0, 4.0))
        pipeline = MagicMock()
        video = MagicMock()
        video.shape = (3, 81, 480, 832)
        pipeline.generate.return_value = video
        wan = SimpleNamespace(WanT2V=MagicMock(return_value=pipeline))
        imageio = MagicMock()
        torch = SimpleNamespace(inference_mode=lambda: lambda function: function)
        payload = dict(spec=vars(VideoSpec('A river', negative_prompt='blur', duration=5,
                            fps=16, resolution='480p', seed=17)), task='t2v-A14B',
                       checkpoint='/models/local', width=832, height=480, output='/tmp/out.mp4')
        location = Path(__file__).resolve().parents[2] / 'inference/workers/wan_worker.py'
        definition = importlib.util.spec_from_file_location('wan_t2v_worker_fixture', location)
        worker = importlib.util.module_from_spec(definition)
        fixtures = {'wan': wan, 'wan.configs': SimpleNamespace(WAN_CONFIGS={'t2v-A14B': config}),
                    'torch': torch, 'imageio': imageio, 'PIL': SimpleNamespace(Image=MagicMock(), ImageOps=MagicMock())}
        with patch.dict(sys.modules, fixtures), patch('sys.stdin', io.StringIO(json.dumps(payload))):
            definition.loader.exec_module(worker)
            worker.main()
        self.assertEqual(wan.WanT2V.call_args.kwargs['checkpoint_dir'], '/models/local')
        pipeline.generate.assert_called_once_with('A river', size=(832, 480), frame_num=81,
            shift=12.0, sample_solver='unipc', sampling_steps=40, guide_scale=(3.0, 4.0),
            n_prompt='blur', seed=17, offload_model=True)
        writer = imageio.get_writer.return_value.__enter__.return_value
        self.assertEqual(writer.append_data.call_count, 80)
        self.assertEqual(imageio.get_writer.call_args.kwargs['fps'], 16)
