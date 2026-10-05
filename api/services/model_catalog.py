"""Reviewed public checkpoints, pinned to immutable Hugging Face revisions.

Sources: https://huggingface.co/{repo_id}/commit/{revision} (2026-10-04).
LLMs use root safetensors/tokenizer assets; media bundles use explicit component
paths. Duplicate representations and repository Python code are excluded.
"""
from dataclasses import dataclass
import re
from pathlib import PurePosixPath
from typing import Literal


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    repo_id: str
    revision: str
    license: str
    estimated_bytes: int
    kind: Literal['llm', 'video', 'speech', 'transcription', 'formatting', 'image'] = 'llm'
    display_name: str | None = None
    layout: Literal['root', 'components', 'single_file'] = 'root'
    subfolder: str = ''
    component_paths: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()
    weight_paths: tuple[str, ...] = ()
    license_url: str | None = None
    license_notice: str | None = None
    inference_available: bool = True


CATALOG: dict[str, CatalogEntry] = {
    entry.id: entry for entry in (
        CatalogEntry('small', 'nvidia/Qwen3.6-35B-A3B-NVFP4',
                     '1355db6a052410cfd62085d94b58866fd0f2c3c5', 'apache-2.0', 23_500_000_000),
        CatalogEntry('medium', 'openai/gpt-oss-120b',
                     'b5c939de8f754692c1647ca79fbf85e8c1e70f8a', 'apache-2.0', 65_000_000_000),
        CatalogEntry('large', 'RedHatAI/GLM-5.3-Flash-NVFP4',
                     '18d55bfd5a2194887738da73753975c9d3842f46', 'mit', 198_000_000_000),
    )
}

# Distilled BF16 only: exclude dev/FP8/NVFP4 and optional DFR duplicate weights.
LTX_FILES = (
    'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors',
    'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
    'vae/ltx-2.5-video-vae-bf16.safetensors',
    'vae/ltx-2.5-audio-vae-bf16.safetensors',
    'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors',
)
CATALOG['ltx-2.5-distilled'] = CatalogEntry(
    'ltx-2.5-distilled', 'Lightricks/LTX-2.5',
    'da86602791e888542b1c755f0b5df376fdb1a3af', 'ltx-2.x-community-license-agreement',
    66 * 1024 ** 3, kind='video', display_name='LTX-2.5 distilled BF16', layout='components',
    required_files=LTX_FILES,
    weight_paths=('diffusion_models', 'text_encoders', 'vae', 'latent_upscale_models'),
    license_url='https://github.com/Lightricks/LTX-2/blob/9ec55f9f22798a3198d9c923856824821bc3317e/LICENSE-2_x',
    license_notice='LTX-2.5 uses the LTX community license. Some commercial uses require a paid license. Hugging Face access approval is required; continuing does not grant access or license rights.',
)

# Original Animate bundle including native pose/detection/SAM2 preprocessing.
WAN_ANIMATE_REVISION = 'cb93a225fbaf1ca100f54e79da8f994995b689b3'
CATALOG['wan22-animate-14b'] = CatalogEntry(
    'wan22-animate-14b', 'Wan-AI/Wan2.2-Animate-14B', WAN_ANIMATE_REVISION,
    'apache-2.0', 58_000_000_000, kind='video', display_name='Wan2.2 Animate-14B',
    layout='components', component_paths=('.', 'google/umt5-xxl'),
    required_files=('config.json', 'configuration.json',
        'diffusion_pytorch_model.safetensors.index.json',
        'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
        'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 'relighting_lora.ckpt',
        'google/umt5-xxl/tokenizer.json', 'google/umt5-xxl/tokenizer_config.json',
        'google/umt5-xxl/special_tokens_map.json', 'google/umt5-xxl/spiece.model',
        'xlm-roberta-large/config.json', 'xlm-roberta-large/tokenizer.json',
        'xlm-roberta-large/sentencepiece.bpe.model',
        'process_checkpoint/det/yolov10m.onnx',
        'process_checkpoint/pose2d/vitpose_h_wholebody.onnx',
        'process_checkpoint/sam2/sam2_hiera_large.pt'),
    weight_paths=('.',),
    license_url=f'https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/{WAN_ANIMATE_REVISION}/README.md',
)

ASSETS = frozenset({
    'config.json', 'configuration.json', 'generation_config.json',
    'hf_quant_config.json', 'model.safetensors.index.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model',
    'vocab.json', 'merges.txt', 'chat_template.jinja', 'preprocessor_config.json',
    'processor_config.json', 'video_preprocessor_config.json', 'LICENSE',
    'LICENSE.txt', 'README.md', 'USAGE_POLICY', 'NOTICE',
    'model_index.json', 'chat_template.json', 'config.yaml', 'metadata.json',
    'added_tokens.json', 'scheduler_config.json',
})


def allowed_asset(name: str, entry: CatalogEntry | None = None) -> bool:
    """Return whether a repository-relative name is an approved root asset.

    Excludes subdirectories, executable code, and duplicate checkpoint formats;
    accepts the known configuration/tokenizer files and root safetensors shards."""
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        return False
    if entry is not None and entry.layout == 'components':
        if name in entry.required_files or name in ('LICENSE', 'LICENSE.txt', 'NOTICE', 'README.md'):
            return True
        if str(path.parent) not in entry.component_paths:
            return False
        name = path.name
        return name in ASSETS or re.fullmatch(
            r'(?:model|diffusion_pytorch_model)(?:-\d+-of-\d+)?\.safetensors(?:\.index\.json)?', name
        ) is not None
    return name in ASSETS or re.fullmatch(
        r'(?:model(?:-\d+-of-\d+|_mtp)?)\.safetensors', name
    ) is not None


def validate_assets(entry: CatalogEntry, filenames: set[str]) -> None:
    """Reject incomplete bundles before downloading or marking them complete."""
    required = set(entry.required_files) if entry.layout == 'components' else {
        'config.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if not required <= filenames:
        raise ValueError('Checkpoint is missing required configuration or weight index')
    weight_paths = entry.weight_paths if entry.layout == 'components' else ('',)
    for prefix in weight_paths:
        if not any(str(PurePosixPath(name).parent) == (prefix or '.')
                   and name.endswith('.safetensors') for name in filenames):
            raise ValueError(f'Checkpoint has no safetensors weights in {prefix or "root"}')
