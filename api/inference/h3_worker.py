"""Optional SGLang runtime entrypoint; never imported by the API process."""
import json
from pathlib import Path
import sys

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult


def accept_result(result, sampling):
    """Accept only upstream-validated output confined to this job's private directory."""
    if not isinstance(result, GenerationResult) or not result.output_file_path:
        raise RuntimeError('H3 generation or audiovisual validation failed')
    root = Path(sampling['output_path']).resolve()
    actual = Path(result.output_file_path)
    if (actual.is_symlink() or actual.resolve().parent != root
            or not actual.is_file() or actual.stat().st_size == 0):
        raise RuntimeError('H3 returned an invalid validated output path')
    expected = root / sampling['output_file_name']
    if actual.resolve() != expected:
        actual.replace(expected)


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
        result = generator.generate(sampling_params_kwargs=payload['sampling'])
        accept_result(result, payload['sampling'])
    finally:
        generator.shutdown()


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
