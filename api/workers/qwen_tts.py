"""Offline Qwen worker, executed only by Kadan in its isolated Python environment."""
import base64
import io
import json
from pathlib import Path
import sys

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


def main():
    """Load a verified local checkpoint and write a PCM WAV after generation completes."""
    payload = json.loads(Path(sys.argv[1]).read_text())
    device = payload['device']
    model = Qwen3TTSModel.from_pretrained(payload['checkpoint'], local_files_only=True,
        device_map=device, dtype=torch.float32 if device == 'cpu' else torch.bfloat16,
        attn_implementation='sdpa')
    waves, rate = generate(model, payload['request'])
    if len(waves) != 1 or rate <= 0 or not len(waves[0]):
        raise ValueError('Qwen returned no complete waveform')
    sf.write(sys.argv[2], waves[0], rate, subtype='PCM_16', format='WAV')


if __name__ == '__main__':
    main()
