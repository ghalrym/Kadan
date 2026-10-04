"""Kadan-owned bounded, cancellable greedy autoregressive loop."""
import torch


def check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('Inference cancelled')


@torch.inference_mode()
def autoregressive_generate(model, tokenizer, messages, device, cancel_event=None,
                            max_new_tokens=256, context_limit=4096):
    check_cancel(cancel_event)
    if not 1 <= max_new_tokens <= 1024:
        raise ValueError('Output token limit must be between 1 and 1024')
    chat = [{'role': item['role'], 'content': item.get('text', item.get('content', ''))} for item in messages]
    tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    if not isinstance(tokens, torch.Tensor):
        tokens = tokens['input_ids']
    if tokens.shape[-1] + max_new_tokens > context_limit:
        raise ValueError(f'Prompt plus output budget exceeds {context_limit} tokens')
    tokens = tokens.to(device)
    eos = model.config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    generated, cache = [], None
    try:
        # Eager attention is quadratic in the query chunk. Bound queries to 32
        # tokens so a full 4096-token prompt never creates a 4096x4096 score map.
        for start in range(0, max(0, tokens.shape[-1] - 32), 32):
            check_cancel(cancel_event)
            output = model(input_ids=tokens[:, start:start + 32], past_key_values=cache,
                           use_cache=True, return_dict=True, logits_to_keep=1)
            cache = output.past_key_values
            del output
        tail_start = ((tokens.shape[-1] - 1) // 32) * 32
        tokens = tokens[:, tail_start:]
        for _ in range(max_new_tokens):
            check_cancel(cancel_event)
            output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                           return_dict=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            value = token.item()
            del output
            if value in stops:
                break
            generated.append(value)
            tokens = token
        check_cancel(cancel_event)
        return tokenizer.decode(generated, skip_special_tokens=True)
    finally:
        del cache
