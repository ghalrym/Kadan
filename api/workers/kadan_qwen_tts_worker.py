"""Offline Qwen worker, executed only by Kadan in its isolated Python environment."""
import base64
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def generate(model, request):
    """Dispatch one complete-waveform generation through the official API."""
    voice = request['voice']
    options = dict(text=request['script'], language=request['language'], non_streaming_mode=True)
    if voice['mode'] == 'describe':
        return model.generate_voice_design(**options, instruct=voice['description'])
    if voice['mode'] == 'custom':
        return model.generate_custom_voice(**options, speaker=voice['speaker'], instruct=voice.get('instruction', ''))
    raw = base64.b64decode(voice['sample'], validate=True)
    audio, rate = sf.read(io.BytesIO(raw), dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return model.generate_voice_clone(**options, ref_audio=(audio, rate),
                                     ref_text=voice.get('transcript'),
                                     x_vector_only_mode=voice.get('speaker_only', False))


def publish(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def serve(root):
    """Load once; process serial requests until the Kadan owner unloads this group."""
    payload = json.loads((root / 'config.json').read_text())
    device = payload['device']
    model = Qwen3TTSModel.from_pretrained(payload['checkpoint'], local_files_only=True,
        device_map=device, dtype=torch.float32 if device == 'cpu' else torch.bfloat16,
        attn_implementation='sdpa')
    publish(root / 'ready.json', {'ok': True})
    while True:
        requests = list(root.glob('*.request.json'))
        if not requests:
            time.sleep(.05)
            continue
        request_path = requests[0]
        payload = json.loads(request_path.read_text())
        request_path.unlink()
        try:
            waves, rate = generate(model, payload['request'])
            if len(waves) != 1 or rate <= 0 or not len(waves[0]):
                raise ValueError('Qwen returned no complete waveform')
            sf.write(payload['output'], waves[0], rate, subtype='PCM_16', format='WAV')
            waves = None
            publish(Path(payload['result']), {'ok': True})
        except Exception:
            publish(Path(payload['result']), {'ok': False})
            raise
        finally:
            # Do not retain cloning samples or generated waveforms while idle.
            payload = waves = None


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-install']:
        for method in ('from_pretrained', 'generate_custom_voice', 'generate_voice_design', 'generate_voice_clone'):
            if not callable(getattr(Qwen3TTSModel, method, None)):
                raise RuntimeError(f'Installed Qwen3-TTS is missing {method}')
        print('Qwen3-TTS worker imports and official generation methods ready')
    elif len(sys.argv) == 3 and sys.argv[1] == '--serve':
        serve(Path(sys.argv[2]))
    else:
        raise SystemExit('Expected --serve DIRECTORY or --check-install')
