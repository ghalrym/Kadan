"""Kadan-owned cancellable decoding with explicit token and memory admission."""
from contextlib import contextmanager
import traceback
import logging
import time
from uuid import uuid4
import torch
from .context import ContextLimitError, ContextMemoryError, estimate_request_memory, resolve_context
from ..resources import ResourceExhausted


log = logging.getLogger(__name__)


def check_cancel(cancel_event):
    """Raise InterruptedError at a cooperative boundary when the optional threading event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('Inference cancelled')


@contextmanager
def request_memory(resources, owner, config, total, prompt, device, cancel_event, expert_headroom_bytes=0):
    """Lease estimated request memory and staging RAM until cleanup. Probe one expert slot without
    promising future capacity; synchronize CUDA before release. Tests may omit resources.
    """
    if resources is None:
        # Only small standalone numerical tests omit the production manager.
        yield
        return
    estimate = estimate_request_memory(config, total, prompt)
    if isinstance(expert_headroom_bytes, bool) or not isinstance(expert_headroom_bytes, int) or expert_headroom_bytes < 0:
        raise ContextMemoryError('Invalid expert transfer headroom')
    # Hold contiguous-copy + pinned staging headroom for the entire request.
    host_bytes = estimate.host_bytes + 2 * expert_headroom_bytes
    device = torch.device(device)
    try:
        reservation = resources.reserve(
            f'{owner}:request:{uuid4()}', 'llm', host_bytes=host_bytes,
            device_bytes={device.index or 0: estimate.device_bytes} if device.type == 'cuda' else {},
            cancel_event=cancel_event)
    except ResourceExhausted as exc:
        raise ContextMemoryError(
            f'Request needs approximately {estimate.device_bytes / 1024**3:.2f} GiB additional GPU memory '
            f'and {host_bytes / 1024**2:.1f} MiB host memory for {total} prompt-plus-output tokens; '
            'it does not fit available RAM/VRAM. '
            'Reduce the request or output budget, or free other workloads. Context was not truncated.'
        ) from exc
    executing = False
    try:
        with reservation.lease(cancel_event):
            # Probe a working expert/tile slot while request memory is protected.
            # This may evict idle cache entries. It is not a future-capacity
            # guarantee: every actual cache transfer still admits independently.
            if expert_headroom_bytes:
                try:
                    headroom = resources.reserve(
                        f'{owner}:expert-headroom:{uuid4()}', 'llm',
                        device_bytes={device.index or 0: expert_headroom_bytes} if device.type == 'cuda' else {},
                        cancel_event=cancel_event)
                except ResourceExhausted as exc:
                    raise ContextMemoryError(
                        f'Request KV/workspace plus a {expert_headroom_bytes / 1024**2:.1f} MiB working expert '
                        'does not fit available memory. Reduce the request or free other workloads; context was not truncated.'
                    ) from exc
                headroom.release()
            executing = True
            yield
    finally:
        if executing and device.type == 'cuda':
            torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        reservation.release()


@torch.inference_mode()
def autoregressive_generate(model, tokenizer, messages, device, cancel_event=None,
                            max_new_tokens=256, context_limit=None, resources=None, owner='inference', expert_headroom_bytes=0, chat_template_kwargs=None):
    """Return greedy decoded text from role/text messages using 32-token prefill chunks. Validate
    context and admit memory before GPU transfer; cancellation and failed forwards release
    request references before the lease ends.
    """
    check_cancel(cancel_event)
    if not 1 <= max_new_tokens <= 1024:
        raise ContextLimitError('Output token limit must be between 1 and 1024')
    _, limit = resolve_context(model.config, context_limit)
    chat = [{'role': item['role'], 'content': item.get('text', item.get('content', ''))} for item in messages]
    tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt', **(chat_template_kwargs or {}))
    if not isinstance(tokens, torch.Tensor):
        tokens = tokens['input_ids']
    prompt = tokens.shape[-1]
    total = prompt + max_new_tokens
    if prompt < 1 or total > limit:
        raise ContextLimitError(
            f'Prompt ({prompt}) plus output budget ({max_new_tokens}) requires {total} tokens; '
            f'effective context is {limit}. Change the model context setting or shorten the request; no content was truncated.'
        )
    eos = model.config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    generated, cache, output, token = [], None, None, None
    started = time.monotonic()
    # Admit before moving tokens to GPU or allocating any request cache/workspace.
    with request_memory(resources, owner, model.config, total, prompt, device, cancel_event, expert_headroom_bytes):
        try:
            tokens = tokens.to(device)
            for start in range(0, max(0, prompt - 32), 32):
                check_cancel(cancel_event)
                output = model(input_ids=tokens[:, start:start + 32], past_key_values=cache,
                               use_cache=True, return_dict=True, logits_to_keep=1)
                cache = output.past_key_values
                output = None
            tail_start = ((prompt - 1) // 32) * 32
            tokens = tokens[:, tail_start:]
            for step in range(max_new_tokens):
                check_cancel(cancel_event)
                output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                               return_dict=True, logits_to_keep=1)
                cache = output.past_key_values
                token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                value = token.item()
                if step == 0:
                    log.info("LLM first token after %.3fs; prompt_tokens=%d", time.monotonic()-started, prompt)
                output = None
                if value in stops:
                    break
                generated.append(value)
                tokens = token
            check_cancel(cancel_event)
            return tokenizer.decode(generated, skip_special_tokens=True)
        except ResourceExhausted as exc:
            traceback.clear_frames(exc.__traceback__)
            raise ContextMemoryError(
                'GPU expert admission failed because available memory changed during generation. '
                'Free other workloads or reduce the request and retry; context was not truncated.'
            ) from exc
        except BaseException as exc:
            # Release failed-forward frame references before releasing admission.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            log.info("LLM generation ended after %.3fs; output_tokens=%d", time.monotonic()-started, len(generated))
            tokens = cache = output = token = None
