"""Verify the exact upstream call contract with tiny tensor fixtures."""
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class LTXWorkerTests(unittest.TestCase):
    def test_upstream_contract_padding_crop_and_duration(self):
        audio_type = lambda **kw: SimpleNamespace(**kw)
        pipeline = Mock(return_value=SimpleNamespace(
            video=np.ones((9, 64, 64, 3)), num_frames=9, tiling_config=None,
            audio=SimpleNamespace(waveform=np.ones((2, 8)), sampling_rate=4)))
        constructor = Mock(return_value=pipeline)
        encoded = {}
        def encode(**kwargs):
            encoded.update(kwargs)
            encoded['video'] = list(kwargs['video'])
        modules = {
            'torch': SimpleNamespace(Tensor=np.ndarray, inference_mode=lambda: lambda function: function, device=lambda value: value),
            'ltx_core.types': SimpleNamespace(Audio=audio_type),
            'ltx_core.model.video_vae': SimpleNamespace(get_video_chunks_number=lambda *_: 1),
            'ltx_pipelines.distilled': SimpleNamespace(DistilledPipeline=constructor),
            'ltx_pipelines.utils.media_io': SimpleNamespace(encode_video=encode),
            'ltx_pipelines.utils.model_paths': SimpleNamespace(ModelPaths=SimpleNamespace(from_split=lambda **kw: kw)),
            'ltx_pipelines.utils.types': SimpleNamespace(OffloadMode=SimpleNamespace(CPU='cpu')),
        }
        payload = dict(spec=dict(prompt='forest', seed=19, duration=1, fps=3),
                       assets={'spatial_upsampler_path': 'up', 'transformer_path': 'weights'},
                       output='output.mp4', width=2, height=2)
        with patch.dict('sys.modules', modules), patch('sys.stdin', io.StringIO(json.dumps(payload))):
            worker = runpy.run_path(str(Path(__file__).parents[2] / 'inference/workers/ltx.py'))
            worker['main']()
        pipeline.assert_called_once_with(prompt='forest', seed=19, height=64, width=64,
            num_frames=9, frame_rate=3, images=[], enhance_prompt=False)
        self.assertEqual(encoded['video'][0].shape, (3, 2, 2, 3))
        self.assertEqual(encoded['audio'].waveform.shape, (2, 4))
        self.assertEqual(constructor.call_args.kwargs['offload_mode'], 'cpu')
