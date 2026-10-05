"""Reviewed public checkpoints, pinned to immutable Hugging Face revisions.

Sources: https://huggingface.co/{repo_id}/commit/{revision} (2026-10-04).
LLMs use root safetensors/tokenizer assets; media bundles use explicit component
paths. Duplicate representations and repository Python code are excluded.
"""
from dataclasses import dataclass
import re
from pathlib import PurePosixPath
from typing import Literal
from api.services.qwen_tts_catalog import SPEECH_MODELS


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

# Official complete checkpoint, including its bundled audio tokenizer.
_speech = SPEECH_MODELS['qwen-tts-0.6b-custom']
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

# Official complete checkpoint, including its bundled audio tokenizer.
_speech = SPEECH_MODELS['qwen-tts-1.7b-design']
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

# Official complete checkpoint, including its bundled audio tokenizer.
_speech = SPEECH_MODELS['qwen-tts-1.7b-base']
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

# Official complete checkpoint, including its bundled audio tokenizer.
_speech = SPEECH_MODELS['qwen-tts-0.6b-base']
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
