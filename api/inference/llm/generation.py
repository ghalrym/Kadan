"""Kadan-owned cancellable decoding with explicit token and memory admission."""
from contextlib import contextmanager
import traceback
import logging
import hashlib
import time
from uuid import uuid4
import torch
from .streaming import TextEvents
from .context import ContextLimitError, ContextMemoryError, estimate_request_memory, resolve_context
from ..resources import ResourceExhausted


log = logging.getLogger('uvicorn.error')


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
                            max_new_tokens=256, context_limit=None, resources=None, owner='inference', expert_headroom_bytes=0, chat_template_kwargs=None, on_event=None, conversation_cache=None, conversation_id=None):
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
    streamer = TextEvents(tokenizer, on_event) if on_event else None
    finish_reason = "length"
    retained = conversation_cache if conversation_id else None
    identity, reused = None, 0
    input_ids = tokens[0].tolist()
    consumed = prompt
    reason = 'disabled'
    success = False
    if retained is not None:
        identity = hashlib.sha256(repr((id(model), limit, max_new_tokens, sorted(str(s) for s in stops),
            getattr(tokenizer, 'chat_template', None), chat_template_kwargs)).encode()).hexdigest()
    first_token_at = last_token_at = None
    prefill_started = None
    started = time.monotonic()
    # Admit before moving tokens to GPU or allocating any request cache/workspace.
    with request_memory(resources, owner, model.config, total, prompt, device, cancel_event, expert_headroom_bytes):
        try:
            if retained is not None:
                cache, reused, reason = retained.restore(conversation_id, input_ids, identity, device,
                                                          prompt - 1, history=chat)
            tokens = tokens.to(device)
            cursor = reused
            prefill_started = time.monotonic()
            while prompt - cursor > 32:
                check_cancel(cancel_event)
                end = min(cursor + 32, prompt)
                output = model(input_ids=tokens[:, cursor:end], past_key_values=cache,
                               use_cache=True, return_dict=True, logits_to_keep=1)
                cache = output.past_key_values
                output = None
                cursor = end
            tokens = tokens[:, cursor:]
            for step in range(max_new_tokens):
                check_cancel(cancel_event)
                output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                               return_dict=True, logits_to_keep=1)
                cache = output.past_key_values
                consumed = prompt + len(generated)
                token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                value = token.item()
                sampled_at = time.monotonic()
                if step == 0:
                    log.info("LLM first token after %.3fs; prompt_tokens=%d", sampled_at-started, prompt)
                output = None
                if value in stops:
                    finish_reason = "stop"
                    break
                last_token_at = sampled_at
                if first_token_at is None:
                    first_token_at = last_token_at
                generated.append(value)
                if streamer is not None:
                    streamer.put(torch.tensor([value]))
                tokens = token
            check_cancel(cancel_event)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if streamer is not None:
                streamer.end()
            if on_event is not None:
                decode_seconds = (last_token_at - first_token_at) if len(generated) > 1 else 0
                on_event({'timing': {
                    'generation_ttft_ms': (first_token_at - started) * 1000 if first_token_at is not None else None,
                    'prefill_ms': (first_token_at - prefill_started) * 1000 if first_token_at is not None else None,
                    'decode_tokens_per_second': (len(generated) - 1) / decode_seconds if decode_seconds > 0 else None,
                    'output_tokens': len(generated), 'prefill_tokens': prompt - reused,
                }})
            if retained is not None:
                # Cache describes inputs already consumed, never the sampled next
                # token. On EOS all answer tokens were consumed; on length stop
                # the last answer token is intentionally left for the next prefill.
                # Hybrid recurrence cannot be cropped or rewound to fix a mismatch.
                completed = chat + [{'role': 'assistant', 'content': text}]
                canonical = tokenizer.apply_chat_template(completed, tokenize=True,
                    add_generation_prompt=False, return_tensors='pt', **(chat_template_kwargs or {}))
                if not isinstance(canonical, torch.Tensor):
                    canonical = canonical['input_ids']
                consumed_ids = (input_ids + generated)[:consumed]
                if canonical[0].tolist()[:consumed] == consumed_ids:
                    retention_reason = retained.capture(conversation_id, consumed_ids, identity, cache, adopt=True)
                else:
                    retained.invalidate(conversation_id)
                    retention_reason = 'noncanonical_completed_tokens'
                check_cancel(cancel_event)
                usage = {'hit': reused > 0, 'reused_tokens': reused, 'reason': reason,
                         **retained.commit(conversation_id, history=completed, retention_reason=retention_reason)}
                usage['prefix_digest'] = hashlib.sha256(repr(consumed_ids).encode()).hexdigest() if usage['stored_tokens'] else None
                log.info('LLM conversation=%s cache=%s', hashlib.sha256(conversation_id.encode()).hexdigest()[:16], usage)
                if on_event is not None:
                    on_event({'cache': usage})
            if on_event is not None:
                on_event({"finish_reason": finish_reason})
            success = True
            return text
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
            if retained is not None and not success:
                retained.invalidate(conversation_id)
