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
class CheckpointSource:
    """An immutable, reviewed Hub source whose selected paths retain their names."""
    repo_id: str
    revision: str
    files: tuple[str, ...] = ()
    component_paths: tuple[str, ...] = ()
    requires_auth: bool = False


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
    requires_auth: bool = False
    source_files: tuple[str, ...] = ()
    auxiliary_sources: tuple[CheckpointSource, ...] = ()


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

# Original-format bundles; do not also download the duplicate Diffusers layout.
H3_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
for family in ('FL2VA',):
    entry = CatalogEntry(
        f'h3-{family.lower()}', 'MiniMaxAI/MiniMax-H3', H3_REVISION,
        'minimax-h3-community-license-agreement', 144_000_000_000,
        kind='video', display_name=f'MiniMax H3 {family}', layout='components',
        subfolder=family,
        component_paths=tuple(f'{family}/{part}' for part in (
            'processor', 'tokenizer', 'text_encoder', 'transformer',
            'audio_vae', 'video_vae', 'video_vae/source')),
        required_files=('model_index.json', f'{family}/model_index.json',
            f'{family}/processor/preprocessor_config.json',
            f'{family}/tokenizer/tokenizer_config.json',
            f'{family}/text_encoder/config.json', f'{family}/transformer/config.json',
            f'{family}/audio_vae/config.json', f'{family}/video_vae/config.json',
            f'{family}/video_vae/source/config.json'),
        weight_paths=tuple(f'{family}/{part}' for part in (
            'text_encoder', 'transformer', 'audio_vae', 'video_vae/source')),
        license_url=f'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/{H3_REVISION}/LICENSE',
        license_notice="MiniMax H3’s license excludes use in the US, EU, UK and South Korea, including personal use. Continuing does not grant rights under the license.",
        inference_available=False,
    )
    CATALOG[entry.id] = entry

# Qwen's complete native Diffusers bundle; assets/ contains documentation images.
QWEN_IMAGE_REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
CATALOG['qwen-image-2.1'] = CatalogEntry(
    'qwen-image-2.1', 'Qwen/Qwen-Image-2.1', QWEN_IMAGE_REVISION,
    'qwen-research', 33_100_000_000, kind='image', display_name='Qwen-Image-2.1',
    layout='components', component_paths=('processor', 'scheduler', 'text_encoder', 'transformer', 'vae'),
    required_files=('model_index.json', 'LICENSE', 'processor/preprocessor_config.json',
        'processor/tokenizer.json', 'processor/tokenizer_config.json', 'processor/chat_template.jinja',
        'scheduler/scheduler_config.json', 'text_encoder/config.json',
        'text_encoder/model.safetensors.index.json', 'transformer/config.json',
        'transformer/diffusion_pytorch_model.safetensors.index.json', 'vae/config.json'),
    weight_paths=('text_encoder', 'transformer', 'vae'),
    license_url=f'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/{QWEN_IMAGE_REVISION}/LICENSE',
    license_notice='Qwen-Image-2.1 is licensed for research and evaluation only. Commercial use requires a separate license; continuing does not grant commercial rights.',
)

CATALOG['flux-klein-4b'] = CatalogEntry(
    'flux-klein-4b', 'black-forest-labs/FLUX.2-klein-4B', 'e7b7dc27f91deacad38e78976d1f2b499d76a294',
    'apache-2.0', 16_000_000_000, kind='image', display_name='FLUX.2 klein 4B',
    layout='components', component_paths=('tokenizer', 'scheduler', 'text_encoder', 'transformer', 'vae'),
    required_files=('model_index.json', 'LICENSE.md', 'tokenizer/tokenizer.json',
        'tokenizer/tokenizer_config.json', 'scheduler/scheduler_config.json',
        'text_encoder/config.json', 'text_encoder/model.safetensors.index.json',
        'transformer/config.json', 'vae/config.json'),
    weight_paths=('text_encoder', 'transformer', 'vae'),
    license_url='https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md',
)

# Official FLUX.2 klein base 9B NVFP4 transformer plus matching pinned companion assets.
CATALOG['flux-klein-base-9b-nvfp4'] = CatalogEntry(
    'flux-klein-base-9b-nvfp4', 'black-forest-labs/FLUX.2-klein-base-9b-nvfp4', 'e651daf0c5d128e5cf7ffeb2da28fca22a8d7467',
    'flux-non-commercial-license', 0, kind='image', display_name='FLUX.2 klein base 9B NVFP4',
    layout='components', source_files=('flux-2-klein-base-9b-nvfp4.safetensors', 'LICENSE.md'),
    requires_auth=True,
    auxiliary_sources=(CheckpointSource('black-forest-labs/FLUX.2-klein-base-9B', '32773329fbe7e81a90ef971740e8ba4b0364ecf3',
        files=('model_index.json', 'transformer/config.json'),
        component_paths=('tokenizer', 'scheduler', 'text_encoder', 'vae'), requires_auth=True),),
    required_files=('flux-2-klein-base-9b-nvfp4.safetensors', 'LICENSE.md', 'model_index.json', 'transformer/config.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json', 'text_encoder/model.safetensors.index.json', 'vae/config.json'),
    weight_paths=('', 'text_encoder', 'vae'),
    license_url='https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4/blob/e651daf0c5d128e5cf7ffeb2da28fca22a8d7467/LICENSE.md',
    license_notice='FLUX.2 klein base 9B NVFP4 has a non-commercial license and usage conditions. Commercial use requires separate rights; continuing does not grant them or approve Hugging Face access.',
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


def _allowed_primary_asset(name: str, entry: CatalogEntry | None = None) -> bool:
    """Return whether a repository-relative name is an approved root asset.

    Excludes subdirectories, executable code, and duplicate checkpoint formats;
    accepts the known configuration/tokenizer files and root safetensors shards."""
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        return False
    if entry is not None and entry.source_files:
        return name in entry.source_files
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


def source_allows(name: str, source: CheckpointSource) -> bool:
    """Match an approved relative path; executable and duplicate formats stay excluded."""
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        return False
    if name in source.files:
        return True
    return str(path.parent) in source.component_paths and (
        path.name in ASSETS or re.fullmatch(
            r'(?:model|diffusion_pytorch_model)(?:-\d+-of-\d+)?\.safetensors(?:\.index\.json)?', path.name) is not None)


def allowed_asset(name: str, entry: CatalogEntry | None = None) -> bool:
    """Accept the union of reviewed primary and auxiliary checkpoint paths."""
    return _allowed_primary_asset(name, entry) or bool(entry and any(
        source_allows(name, source) for source in entry.auxiliary_sources))


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
