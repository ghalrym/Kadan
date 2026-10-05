"""Kadan-owned, offline speech workers with shared memory admission."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid
import wave

from api.inference.resources import ResourceCancelled
from api.services.model_downloads import model_manager as manager
from api.services.qwen_tts_catalog import SPEECH_MODELS
from api.services.speech_enabled import ENABLED_SPEECH_MODELS
from api.services.runtime import runtime_manager


class SpeechUnavailable(RuntimeError):
    pass


def generate_speech(request: dict, cancel: threading.Event) -> dict:
    """Hold RAM/VRAM ownership until the isolated child exits, including cancellation."""
    model = SPEECH_MODELS[request['model_id']]
    if model.id not in ENABLED_SPEECH_MODELS:
        raise SpeechUnavailable('This Qwen3-TTS checkpoint integration is not enabled.')
    python = os.environ.get('KADAN_QWEN_TTS_PYTHON')
    if not python or not Path(python).is_file():
        raise SpeechUnavailable('Configure the isolated Qwen3-TTS worker environment before generating speech.')
    resolver = getattr(manager, 'get_checkpoint', None)
    if resolver is None:
        raise SpeechUnavailable('The shared checkpoint catalog integration is required for Qwen3-TTS.')
    try:
        entry, checkpoint = resolver(model.id)
    except ValueError as exc:
        raise SpeechUnavailable(str(exc)) from exc
    if entry.revision != model.revision:
        raise SpeechUnavailable('The selected Qwen checkpoint revision does not match the provider.')
    resources = runtime_manager.ensure_resources()
    device = os.environ.get('KADAN_QWEN_TTS_DEVICE', 'cuda:0')
    if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise SpeechUnavailable('Qwen speech device must be cpu or cuda:N.')
    # Conservative admission budgets include checkpoint conversion and generation;
    # these are estimates, not measured hardware performance guarantees.
    budget = model.estimated_bytes * 3 + 2 * 1024**3
    devices = {} if device == 'cpu' else {int(device[5:]): budget}
    reservation = resources.reserve('speech-' + uuid.uuid4().hex, 'speech',
        host_bytes=budget, device_bytes=devices, cancel_event=cancel)
    try:
        with reservation.lease(cancel):
            with tempfile.TemporaryDirectory(prefix='kadan-speech-') as directory:
                root = Path(directory)
                input_path, output_path = root / 'input.json', root / 'output.wav'
                input_path.write_text(json.dumps(dict(request=request, checkpoint=str(checkpoint), device=device)))
                env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
                worker = Path(__file__).parents[1] / 'workers' / 'kadan_qwen_tts_worker.py'
                with (root / 'worker.log').open('wb') as log:
                    process = subprocess.Popen([python, str(worker), str(input_path), str(output_path)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env)
                    deadline = time.monotonic() + 1800
                    try:
                        while process.poll() is None:
                            if cancel.wait(.05):
                                raise ResourceCancelled('Speech generation cancelled')
                            if time.monotonic() > deadline:
                                raise SpeechUnavailable('Speech generation timed out.')
                        if process.returncode:
                            raise SpeechUnavailable('Qwen3-TTS worker failed. Check its environment and local checkpoint.')
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                if cancel.is_set():
                    raise ResourceCancelled('Speech generation cancelled')
                try:
                    raw = output_path.read_bytes()
                    with wave.open(io.BytesIO(raw)) as audio:
                        duration = audio.getnframes() / audio.getframerate()
                    if duration <= 0:
                        raise ValueError('Empty audio')
                except (OSError, EOFError, wave.Error, ValueError) as exc:
                    raise SpeechUnavailable('Qwen3-TTS returned invalid audio.') from exc
                return dict(voice=model.name, meta='WAV', script=request['script'],
                    time=f'{duration:.1f}s', audio_base64=base64.b64encode(raw).decode('ascii'),
                    mime_type='audio/wav')
    finally:
        reservation.release()
