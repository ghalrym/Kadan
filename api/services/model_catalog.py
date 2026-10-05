"""Reviewed public checkpoints, pinned to immutable Hugging Face revisions.

Sources: https://huggingface.co/{repo_id}/commit/{revision} (2026-10-04).
LLMs use root safetensors/tokenizer assets; media bundles use explicit component
paths. Duplicate representations and repository Python code are excluded.
"""
from dataclasses import dataclass
import re
from pathlib import PurePosixPath
from typing import Literal

from api.services.whisper_catalog import CHECKPOINTS


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
    asset_url: str | None = None
    asset_name: str | None = None


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

CATALOG['s1-mini'] = CatalogEntry(
    's1-mini', 'superwhisper/s1-mini', '88f6b15896c73bbb13a3b596e0afe8ea0d5150b4',
    'apache-2.0-with-naming-clause', 1_520_000_000, kind='formatting',
    display_name='S1-mini by Superwhisper', layout='components', component_paths=('.',),
    required_files=('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                    'chat_template.jinja', 'LICENSE', 'NOTICE'), weight_paths=('',),
    license_url='https://huggingface.co/superwhisper/s1-mini/blob/88f6b15896c73bbb13a3b596e0afe8ea0d5150b4/LICENSE',
)

# Only independently registered Whisper checkpoints enter the shared catalog.
for whisper in CHECKPOINTS.values():
    entry = CatalogEntry(f'whisper-{whisper.name}', 'openai/whisper', whisper.sha256,
        'mit', 0, kind='transcription', display_name=f'Whisper {whisper.name}',
        layout='single_file', asset_url=whisper.url, asset_name=f'{whisper.name}.pt',
        required_files=(f'{whisper.name}.pt',),
        license_url='https://github.com/openai/whisper/blob/main/LICENSE')
    CATALOG[entry.id] = entry

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
    if entry is not None and entry.layout == 'single_file':
        return name == entry.asset_name and len(path.parts) == 1
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
    if entry.layout == 'single_file':
        if not entry.asset_name or filenames != {entry.asset_name}:
            raise ValueError('Checkpoint is missing its required single file')
        return
    required = set(entry.required_files) if entry.layout == 'components' else {
        'config.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if not required <= filenames:
        raise ValueError('Checkpoint is missing required configuration or weight index')
    weight_paths = entry.weight_paths if entry.layout == 'components' else ('',)
    for prefix in weight_paths:
        if not any(str(PurePosixPath(name).parent) == (prefix or '.')
                   and name.endswith('.safetensors') for name in filenames):
            raise ValueError(f'Checkpoint has no safetensors weights in {prefix or "root"}')
