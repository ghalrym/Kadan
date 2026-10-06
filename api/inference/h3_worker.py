"""Bundled native H3 runtime entrypoint; isolated from API/LLM dependencies."""
import json
import os
from pathlib import Path
import sys



def accept_result(result, sampling):
    """Accept only upstream-validated output confined to this job's private directory."""
    # Spawned schedulers initialize their platform before runtime imports.
    from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult

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


def require_turbo(generator):
    """Fail closed when the native pipeline found no active unmerged adapter.

    SGLang's active map requires at least one enabled LoRA layer. Its loader can
    otherwise merely warn about zero matches and continue with base weights.
    """
    active = generator.list_loras().get('active', {}).get('transformer', [])
    if (len(active) != 1 or active[0].get('merged') is not False
            or active[0].get('strengths') != [1.0]):
        raise RuntimeError('H3 Turbo must be active in dynamic mode at strength 1')


def run(payload):
    """Load pinned serialized INT8 tensors, with dynamic Turbo and native CPU offload."""
    # SGLang spawn workers re-execute this module; importing at module scope
    # constructs backend classes before their platform/plugins are initialized.
    toolkit = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages/nvidia/cu13'
    if (toolkit / 'bin/nvcc').is_file():
        os.environ.setdefault('CUDA_HOME', str(toolkit))
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import MiniMaxH3PipelineConfig

    root = Path(payload['checkpoint'])
    count = payload['num_gpus']
    config = MiniMaxH3PipelineConfig()
    # Serialized Comfy weights are already concatenated Q/K/V; the original
    # release's grouped-to-concatenated conversion must not run a second time.
    config.dit_config.arch_config.qkv_checkpoint_grouped = False
    generator = DiffGenerator.from_pretrained(
        local_mode=True, model_path=str(root / 'FL2VA'), backend='sglang',
        pipeline_config=config,
        num_gpus=count, tp_size=count, ulysses_degree=1,
        component_weights_paths={
            'transformer': str(root / 'FL2VA/transformer/model.safetensors'),
            'text_encoder': str(root / 'FL2VA/text_encoder/model.safetensors'),
            'video_vae': str(root / 'FL2VA/video_vae/model.safetensors'),
            'audio_vae': str(root / 'FL2VA/audio_vae/model.safetensors'),
        },
        component_precisions={'text_encoder': 'fp16',
                              'video_vae': 'fp16', 'audio_vae': 'fp32'},
        lora_path=str(root / 'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'),
        lora_merge_mode='dynamic', lora_scale=1.0,
        cpu_offload_components=['video_vae', 'audio_vae'],
        layerwise_resident_layers={'text_encoder': 0},
        layerwise_prefetch_size={'text_encoder': 1},
        attention_backend='torch_sdpa',
        performance_mode='memory', layerwise_offload_components=['dit', 'text_encoder'],
        dit_offload_prefetch_size=1, dit_layerwise_resident_layers=0,
        enable_torch_compile=False,
    )
    try:
        require_turbo(generator)
        result = generator.generate(sampling_params_kwargs=payload['sampling'])
        accept_result(result, payload['sampling'])
    finally:
        generator.shutdown()


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
