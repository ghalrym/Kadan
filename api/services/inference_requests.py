"""Checkpoint acquisition, media transport, and native-result presentation."""
from __future__ import annotations
import base64
import binascii
import io
import json
import os
from pathlib import Path
import secrets
import struct
import wave
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from api.pydantic_models.inference import (Feature, Operation, NativePayload, ChatInput, DecisionInput,
    ImageInput, SpeechInput, VideoInput, VideoPaths, EmptyPayload, ChatPayload,
    DecisionPayload, ImagePayload, SpeechPayload, TranscriptionPayload, VideoPayload)

from api.inference.errors import InferenceFailure
from api.inference.decisions.laya_answers import parse_answer
from api.services.model_downloads import model_manager
from api.services.native_assets import prepare_assets
from api.inference.stt.catalog import checkpoint as whisper_checkpoint

# Route models stay route-owned; importing them at runtime would create a
# route -> service -> route cycle. These imports describe the service boundary.
if TYPE_CHECKING:
    from api.routes.model_lifecycle import ModelLoadRequest
    from api.routes.v1.chat.completions import CompletionRequest
    from api.routes.v1.decisions import DecisionRequest
    from api.routes.v1.images.generations import ImageRequest
    from api.routes.v1.audio.speech import SpeechRequest
    from api.routes.v1.audio.transcriptions import TranscriptionRequest
    from api.routes.v1.videos.generations import VideoGenerationRequest
    InferenceBody = ModelLoadRequest | CompletionRequest | DecisionRequest | ImageRequest | SpeechRequest | TranscriptionRequest | VideoGenerationRequest

IMAGE_SIZES = {'1:1': (2048, 2048), '4:3': (2400, 1792), '3:4': (1792, 2400), '16:9': (2752, 1536)}


class PreparedRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    payload: NativePayload
    checkpoint_id: str
    context_limit: int | None = None
    supported_context: int | None = None
    seed: int | None = None


def pcm_audio(audio: str) -> bytes:
    if not audio.startswith('data:audio/wav;base64,') or len(audio) > 2 * 1024**2:
        raise InferenceFailure('Supply at most 30 seconds of mono 16-kHz PCM WAV.', 422)
    try:
        raw = base64.b64decode(audio.split(',', 1)[1], validate=True)
        with wave.open(io.BytesIO(raw), 'rb') as source:
            count = source.getnframes()
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, 'NONE') or not 0 < count <= 480000:
                raise ValueError('Expected 1..480000 mono 16-bit samples at 16000 Hz')
            frames = source.readframes(count)
            if len(frames) != count * 2:
                raise ValueError('Truncated WAV')
        return b''.join(struct.pack('<f', value / 32768) for (value,) in struct.iter_unpack('<h', frames))
    except (ValueError, EOFError, wave.Error, binascii.Error) as error:
        raise InferenceFailure(f'Invalid audio: {error}', 422) from error


def prepare_request(feature: Feature, operation: Operation, model: str, body: InferenceBody, workspace: Path, cancel) -> PreparedRequest:
    if operation == 'unload':
        return PreparedRequest(payload=EmptyPayload(), checkpoint_id=model)
    if feature == 'stt':
        return prepare_transcription(model, body, workspace, cancel)

    if feature == 'llm' and not model_manager._language_entry(model).inference_available:
        raise InferenceFailure('This checkpoint has no supported native language engine.', 422)
    entry, checkpoint = model_manager.ensure_checkpoint(model, cancel)
    if feature == 'llm':
        return prepare_chat(entry, checkpoint, body, workspace, cancel, operation)
    elif feature == 'decisions':
        return prepare_decisions(entry, checkpoint, body, workspace, cancel)
    elif feature == 'image':
        return prepare_image(entry, checkpoint, body, workspace, cancel)
    elif feature == 'tts':
        return prepare_speech(entry, checkpoint, body, workspace, cancel)
    elif feature == 'video':
        return prepare_video(entry, checkpoint, body, workspace, cancel)
    raise InferenceFailure('Unsupported inference feature.', 422)


def prepare_transcription(model: str, body: TranscriptionRequest, workspace: Path, cancel) -> PreparedRequest:
    if body.language not in (None, 'en', 'english', 'English'):
        raise InferenceFailure('The native Whisper checkpoint supports English.', 422)
    root = Path(os.getenv('KADAN_NATIVE_WHISPER_MODEL_ROOT', str(model_manager.root / 'native/whisper' / model)))
    assets = Path(os.getenv('KADAN_NATIVE_WHISPER_ASSET_ROOT', str(root / 'assets')))
    if not root.exists():
        entry = whisper_checkpoint(model)
        _, source = model_manager.ensure_checkpoint('whisper-' + entry.name, cancel)
        prepare_assets('whisper', source / (entry.name + '.pt'), root, cancel, entry.sha256)
    for path in (root / 'model.safetensors' , root / 'dimensions.txt', assets / 'english.tokens'):
        if not path.is_file():
            raise InferenceFailure(f'Native Whisper export is incomplete: {path}')
    input_path = workspace / 'pcm.f32'
    input_path.write_bytes(pcm_audio(body.audio))
    return PreparedRequest(checkpoint_id=model, payload=TranscriptionPayload(checkpoint=str(root), assets=str(assets), input=str(input_path)))


def prepare_chat(entry, checkpoint: Path, body: ModelLoadRequest | CompletionRequest, workspace: Path, cancel, operation: Operation) -> PreparedRequest:
    if not entry.inference_available:
        raise InferenceFailure('This checkpoint has no supported native language engine.', 422)
    config = json.loads((checkpoint / 'config.json').read_text())
    text_config = config.get('text_config', config)
    supported = text_config['max_position_embeddings']
    configured = (body.context_limit if operation == 'load' and 'context_limit' in body.model_fields_set
                  else model_manager.configured_context(entry.id))
    effective = supported if configured is None else configured
    if type(effective) is not int or not 1 <= effective <= supported:
        raise InferenceFailure('Configured context exceeds the checkpoint limit.', 422)
    input_path = None
    if operation != 'load':
        input_path = workspace / 'chat.json'
        input_path.write_text(ChatInput(messages=body.messages).model_dump_json(), encoding='utf-8')
    return PreparedRequest(checkpoint_id=entry.id, context_limit=effective, supported_context=supported,
        payload=ChatPayload(checkpoint=str(checkpoint), architecture=config['model_type'],
            context_limit=effective, input=str(input_path) if input_path is not None else None))


def prepare_decisions(entry, checkpoint: Path, body: DecisionRequest, workspace: Path, cancel) -> PreparedRequest:
    return PreparedRequest(checkpoint_id=entry.id, payload=DecisionPayload(
        checkpoint=str(checkpoint), request=DecisionInput(state=body.state, questions=body.questions)))


def prepare_image(entry, checkpoint: Path, body: ImageRequest, workspace: Path, cancel) -> PreparedRequest:
    width, height = IMAGE_SIZES[body.aspect]
    seed = body.seed if body.seed is not None else secrets.randbits(32)
    return PreparedRequest(checkpoint_id=entry.id, seed=seed, payload=ImagePayload(
        checkpoint=str(checkpoint), output=str(workspace), request=ImageInput(
            prompt=body.prompt, width=width, height=height, steps=body.steps, seed=seed, count=body.count)))


def prepare_speech(entry, checkpoint: Path, body: SpeechRequest, workspace: Path, cancel) -> PreparedRequest:
    tokenizer = Path(os.getenv('KADAN_NATIVE_TTS_TOKENIZER', str(model_manager.root / 'native/tts/tokenizer.json')))
    if not tokenizer.is_file():
        prepare_assets('tts', checkpoint, tokenizer.parent, cancel)
    return PreparedRequest(checkpoint_id=entry.id, payload=SpeechPayload(
        checkpoint=str(checkpoint), tokenizer=str(tokenizer), output=str(workspace / 'audio.wav'),
        request=SpeechInput(script=body.script, speaker=body.voice.speaker, language=body.language,
                           instruction=body.voice.instruction)))


def prepare_video(entry, checkpoint: Path, body: VideoGenerationRequest, workspace: Path, cancel) -> PreparedRequest:
    names = dict(tokenizer='FL2VA/tokenizer/tokenizer.json', text='FL2VA/text_encoder/model.safetensors',
        denoiser='FL2VA/transformer/model.safetensors', turbo='loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
        vae='FL2VA/video_vae/model.safetensors', audio_vae='FL2VA/audio_vae/model.safetensors')
    return PreparedRequest(checkpoint_id=entry.id, payload=VideoPayload(
        paths=VideoPaths(**{key: str(checkpoint / name) for key, name in names.items()}),
        output=str(workspace / 'video.mp4'), request=VideoInput(prompt=body.prompt,
            short_edge=int(body.resolution[:-1]), aspect=body.aspect, duration=body.duration, seed=body.seed)))


def read_artifact(path: Path, workspace: Path, maximum: int) -> bytes:
    if path.parent != workspace or path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise InferenceFailure('Native worker returned an invalid output artifact.', 502)
    return path.read_bytes()


def present_result(feature: Feature, model: str, body: InferenceBody, prepared: PreparedRequest,
                   result: dict, workspace: Path, job_id: str):
    if feature == 'llm':
        return {**result, 'model': model, 'cache': dict(hit=False, reused_tokens=0, stored_tokens=0,
            host_bytes=0, device_bytes={}, reason='native_prefix_reuse_unavailable',
            retention_reason='disabled', limit_bytes=0) if body.conversation_id else None}
    if feature == 'decisions':
        answers = result.get('answers')
        if not isinstance(answers, list) or len(answers) != len(body.questions):
            raise InferenceFailure('Native decision answer count is invalid.', 502)
        output = []
        for question, answer in zip(body.questions, answers):
            if answer['key'] != question.key or answer['type'] != question.type:
                raise InferenceFailure('Native decision answer order is invalid.', 502)
            value = parse_answer(question, dict(type=question.type.lower(),
                **{question.type.lower(): answer['value']}, answer_confidence=answer['confidence'],
                probabilities=answer['probabilities']))
            output.append(value.model_dump(mode='json'))
        return output
    if feature == 'image':
        width, height = IMAGE_SIZES[body.aspect]
        if len(result['images']) != body.count:
            raise InferenceFailure('Native image output count is invalid.', 502)
        images = [base64.b64encode(read_artifact(Path(path), workspace, width * height * 4 + 1024**2)).decode('ascii')
                  for path in result['images']]
        return {'image': dict(id=job_id, mode='Generate', prompt=body.prompt,
            aspect={'1:1': 'square', '4:3': 'landscape', '3:4': 'portrait', '16:9': 'wide'}[body.aspect],
            seeds=[(prepared.seed + index) % 2**64 for index in range(body.count)],
            meta=f'{width}×{height} · {body.steps} steps · native CUDA', images_base64=images, mime_type='image/png')}
    if feature == 'tts':
        audio = read_artifact(Path(result['output']), workspace, 4 * 1024**2)
        with wave.open(io.BytesIO(audio)) as source:
            duration = source.getnframes() / source.getframerate()
        return dict(voice=body.voice.speaker, meta='WAV', script=body.script, time=f'{duration:.1f}s',
            audio_base64=base64.b64encode(audio).decode('ascii'), mime_type='audio/wav')
    if feature == 'stt':
        return dict(text=result['text'], raw_text=result['text'], language='en', model=model,
            formatting_status='unavailable' if body.formatting else 'disabled')
    return result
