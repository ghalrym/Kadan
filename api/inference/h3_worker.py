"""Optional SGLang runtime entrypoint; never imported by the API process."""
import json
from pathlib import Path
import sys

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


def run(payload):
    """Load the original FL2VA bundle locally and write joint audio/video output."""
    generator = DiffGenerator.from_pretrained(
        local_mode=True, model_path=payload['checkpoint'], model_variant='fl2va',
        num_gpus=1, tp_size=1, ulysses_degree=1,
        performance_mode='memory', layerwise_offload_components=['dit', 'text_encoder'],
        dit_offload_prefetch_size=1, dit_layerwise_resident_layers=0,
        enable_torch_compile=False,
    )
    try:
        generator.generate(sampling_params_kwargs=payload['sampling'])
    finally:
        generator.shutdown()


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
