"""Reviewed public checkpoints, pinned to immutable Hugging Face revisions.

Sources: https://huggingface.co/{repo_id}/commit/{revision} (2026-10-04).
LLMs use root safetensors/tokenizer assets; media bundles use explicit component
paths. Duplicate representations and repository Python code are excluded.
"""
from dataclasses import dataclass
import re
import json
import hashlib
from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal
from api.inference.tts.catalog import SPEECH_MODELS
from api.inference.decisions.laya_checkpoint import DEFAULT_MODEL, DEFAULT_REVISION

from api.inference.stt.catalog import get_whisper_checkpoints


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    repo_id: str
    revision: str
    license: str
    estimated_bytes: int
    kind: Literal['llm', 'video', 'speech', 'transcription', 'formatting', 'image', 'decision'] = 'llm'
    display_name: str | None = None
    layout: Literal['root', 'components', 'single_file', 'composite'] = 'root'
    subfolder: str = ''
    component_paths: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()
    weight_paths: tuple[str, ...] = ()
    license_url: str | None = None
    license_notice: str | None = None
    inference_available: bool = True
    asset_url: str | None = None
    asset_name: str | None = None
    manifest: tuple[dict, ...] = ()


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

# A content-addressed composite: native release configs plus serialized INT8
# tensors, FP16 video VAE, FP32 audio VAE, and the tested Turbo adapter.
_H3_MANIFEST_BYTES = Path(__file__).with_name('h3_int8_manifest.json').read_bytes()
H3_INT8_REVISION = hashlib.sha256(_H3_MANIFEST_BYTES).hexdigest()
H3_INT8_MANIFEST = tuple(json.loads(_H3_MANIFEST_BYTES))
CATALOG['h3-fl2va-int8-turbo'] = CatalogEntry(
    'h3-fl2va-int8-turbo', 'Comfy-Org/MiniMax-H3', H3_INT8_REVISION,
    'minimax-h3-community-license-agreement', sum(f['size'] for f in H3_INT8_MANIFEST),
    kind='video', display_name='MiniMax H3 FL2VA INT8 + Turbo', layout='composite',
    required_files=tuple(f['name'] for f in H3_INT8_MANIFEST),
    manifest=H3_INT8_MANIFEST,
    license_url=CATALOG['h3-fl2va'].license_url,
    license_notice=CATALOG['h3-fl2va'].license_notice,
)

# Only independently registered Whisper checkpoints enter the shared catalog.
for whisper in get_whisper_checkpoints().values():
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
    if entry is not None and entry.layout == 'composite':
        return name in entry.required_files
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
    if entry.layout == 'composite':
        if filenames != set(entry.required_files):
            raise ValueError('Composite checkpoint does not match its pinned manifest')
        return
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

# Official complete checkpoint, including its bundled audio tokenizer.
_speech = SPEECH_MODELS['qwen-tts-1.7b-custom']
CATALOG[_speech.id] = CatalogEntry(
    _speech.id, 'Qwen/' + _speech.name, _speech.revision, 'apache-2.0',
    _speech.estimated_bytes, kind='speech', display_name=_speech.name,
    layout='components', component_paths=('.', 'speech_tokenizer'),
    required_files=('config.json', 'generation_config.json', 'tokenizer_config.json',
        'preprocessor_config.json', 'merges.txt', 'vocab.json', 'model.safetensors',
        'speech_tokenizer/config.json', 'speech_tokenizer/configuration.json',
        'speech_tokenizer/preprocessor_config.json', 'speech_tokenizer/model.safetensors'),
    weight_paths=('', 'speech_tokenizer'),
)

# Complete original image checkpoint consumed by the native executor.
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

# The C++ Laya executor consumes these four files directly; no encoder export.
CATALOG['laya'] = CatalogEntry(
    'laya', DEFAULT_MODEL, DEFAULT_REVISION, 'unknown', 0,
    kind='decision', display_name='Laya', layout='components',
    required_files=('rl_agent_config.json', 'model.safetensors',
                    'tokenizer/tokenizer.json', 'encoder/config.json'),
    weight_paths=('',),
)
