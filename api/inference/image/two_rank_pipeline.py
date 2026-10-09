"""Request-scoped adapter around the numerically validated fixed Qwen split.

Optional GPU imports live in this worker-only module, never API startup.
"""
from functools import wraps
import logging
import time

import torch

from api.inference.image.sequence_parallel_transformer import SequenceParallelQwenTransformer, validate_transformer_output


logger = logging.getLogger(__name__)


class TwoRankQwenPipeline:
    def __init__(self, pipeline, rank, control, adapter_factory=SequenceParallelQwenTransformer):
        self.pipeline, self.rank, self.control = pipeline, rank, control
        self.adapter_factory = adapter_factory

    def generate(self, job, prompt, seed):
        pipeline = self.pipeline
        # A scheduler carries step state. Reconstruct it before every independent
        # request, including A -> B -> A, rather than trusting reset side effects.
        pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        adapter = self.adapter_factory(pipeline.transformer, self.rank, self.control, job)
        original = pipeline.transformer.forward
        postprocess = pipeline.image_processor.postprocess
        step = 0
        started = time.monotonic()
        @wraps(original)
        def forward(*args, **kwargs):
            nonlocal step
            mode = kwargs['kv_cache_mode']
            if mode != ('extract' if step == 0 else 'cached') or step >= 40:
                raise ValueError('Unexpected fixed-shape image step/cache mode')
            adapter.mode, adapter.step = mode, step
            step += 1
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 1:
                raise ValueError('Unexpected image transformer result')
            validated = validate_transformer_output(adapter.gather(result[0]), mode)
            logger.info('image_step_returned job=%s rank=%s step=%s monotonic=%.6f elapsed_seconds=%.6f',
                        job, self.rank, step, time.monotonic(), time.monotonic()-started)
            return (validated,)
        @wraps(postprocess)
        def finite_output(image, *args, **kwargs):
            if not bool(torch.isfinite(image).all()):
                raise ValueError('Nonfinite VAE output before normalization/clipping')
            return postprocess(image, *args, **kwargs)
        try:
            adapter.install()
            pipeline.transformer.forward = forward
            pipeline.image_processor.postprocess = finite_output
            with torch.no_grad():
                output = pipeline(prompt=prompt, width=2048, height=2048,
                    num_inference_steps=40, num_images_per_prompt=1, generator=generator,
                    true_cfg_scale=1.0, use_kv_cache=True, output_type='pil')
            if step != 40 or len(output.images) != 1:
                raise ValueError('Incomplete image trajectory')
            image = output.images[0]
            if image.size != (2048, 2048) or image.mode != 'RGBA':
                raise ValueError('Expected one 2048-square RGBA image')
            return image, time.monotonic()-started
        finally:
            pipeline.transformer.forward = original
            pipeline.image_processor.postprocess = postprocess
            adapter.close()  # Drop every request prefix on success and failure.
            # Scheduler timesteps/sigmas are not module buffers: pipeline.to('cpu')
            # cannot move them when the next modality parks this pipeline.
            pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
            pipeline._current_timestep = None
