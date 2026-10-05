"""Local English transcript normalization with S1-mini by Superwhisper."""
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import re
import threading
import traceback

from api.services.runtime import runtime_manager

MODEL_ID = 'superwhisper/s1-mini'
REVISION = '88f6b15896c73bbb13a3b596e0afe8ea0d5150b4'
ATTRIBUTION = 'S1-mini by Superwhisper'
RAM_BYTES = 6 * 1024 ** 3
SYSTEM = (
    'You are a text normalizer for speech-to-text transcripts. The input begins '
    'with a control line specifying the styling, structure, and context settings; '
    'clean the transcript to match those settings and output only the cleaned text.'
)
CONTROL = '[Styling: semi-formal] [Structure: prose] [Context: general]'


@dataclass(frozen=True)
class FormattedTranscript:
    raw_text: str
    text: str
    formatting_status: str
    formatting_model: str | None = None


def transcript_chunks(text, tokenizer, limit=1000):
    """Prefer sentence boundaries; split oversized sentences without dropping characters."""
    pending = ''
    for sentence in re.split(r'(?<=[.!?])(?=\s)', text):
        if pending and len(tokenizer.encode(pending + sentence, add_special_tokens=False)) > limit:
            yield pending
            pending = ''
        while len(tokenizer.encode(sentence, add_special_tokens=False)) > limit:
            low, high = 1, len(sentence)
            while low < high:
                middle = (low + high + 1) // 2
                if len(tokenizer.encode(sentence[:middle], add_special_tokens=False)) <= limit:
                    low = middle
                else:
                    high = middle - 1
            boundary = sentence.rfind(' ', 0, low)
            cut = boundary + 1 if boundary > 0 else low
            yield sentence[:cut]
            sentence = sentence[cut:]
        pending += sentence
    if pending:
        yield pending


class NativeNormalizer:
    def __init__(self, tokenizer, model, torch):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch

    def normalize(self, text):
        """Use the trained prompt and greedy decoding; reject truncated output."""
        results = []
        for chunk in transcript_chunks(text, self.tokenizer):
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': CONTROL + '\n' + chunk}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(prompt, return_tensors='pt')
            budget = math.ceil(len(self.tokenizer.encode(chunk, add_special_tokens=False)) * 1.3) + 32
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
            tokens = output[0][inputs['input_ids'].shape[-1]:]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if len(tokens) >= budget and int(tokens[-1]) not in eos:
                raise ValueError('S1-mini output exceeded the chunk budget')
            results.append(self.tokenizer.decode(tokens, skip_special_tokens=True).strip())
        return ' '.join(result for result in results if result)

    def close(self):
        """Free model and tokenizer before returning the reservation."""
        self.model = self.tokenizer = None
        gc.collect()


def load_normalizer():
    """Load only an already provisioned checkpoint; never fetch weights during inference."""
    # Optional heavyweight imports are delayed so missing native packages do not
    # prevent API startup or raw transcription when formatting is disabled.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(os.environ.get('KADAN_MODEL_DIR', '~/.local/share/kadan/models')).expanduser()
    path = Path(os.environ.get('KADAN_S1_MODEL_DIR', str(root / f's1-mini-{REVISION}'))).expanduser()
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                 'chat_template.jinja', 'LICENSE', 'NOTICE'):
        if not (path / name).is_file():
            raise ValueError('S1-mini checkpoint is incomplete')
    # Published weights are 1.50 GB; reserve FP32 CPU weights, mapping and workspace.
    if (path / 'model.safetensors').stat().st_size > 1_600_000_000:
        raise ValueError('S1-mini checkpoint exceeds the supported size')
    config = json.loads((path / 'config.json').read_text())
    expected = dict(model_type='qwen3', hidden_size=1024, intermediate_size=3072,
                    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                    head_dim=128, vocab_size=151936, tie_word_embeddings=True)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported S1-mini architecture')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, device_map=None).eval()
    return NativeNormalizer(tokenizer, model, torch)


class TranscriptFormatter:
    def __init__(self, loader=load_normalizer, resources=None):
        self.loader, self.resources = loader, resources
        self._lock = threading.Lock()
        self._owner = f's1-mini:{id(self)}'

    def format(self, raw_text, language, enabled=True):
        """Preserve ASR output on disabled, unsupported, busy or failed formatting."""
        def result(text, status, model=None):
            return FormattedTranscript(raw_text, text, status, model)
        if not enabled:
            return result(raw_text, 'disabled')
        if not language or language.lower().split('-')[0] not in ('en', 'english'):
            return result(raw_text, 'unsupported_language')
        if not raw_text.strip():
            return result(raw_text, 'empty')
        if not self._lock.acquire(blocking=False):
            return result(raw_text, 'busy')
        reservation = normalizer = None
        try:
            resources = self.resources or runtime_manager.ensure_resources()
            reservation = resources.reserve(self._owner, 'speech', host_bytes=RAM_BYTES)
            with reservation.lease():
                try:
                    normalizer = self.loader()
                    text = normalizer.normalize(raw_text)
                    if not isinstance(text, str):
                        raise ValueError('Invalid S1-mini output')
                    return result(text, 'formatted', ATTRIBUTION)
                finally:
                    if normalizer is not None:
                        normalizer.close()
                        normalizer = None
        except Exception as exc:
            # Failed loader/generator frames can retain tensors after unwinding.
            current = exc
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                traceback.clear_frames(current.__traceback__)
                current = current.__cause__ or current.__context__
            gc.collect()
            return result(raw_text, 'unavailable')
        finally:
            if reservation is not None:
                reservation.release()
            self._lock.release()


transcript_formatter = TranscriptFormatter()


def format_transcript(raw_text, language, enabled=True):
    """Apply the configured formatting preference after the ASR lease is released."""
    return transcript_formatter.format(raw_text, language, enabled)
