"""Bounded BF16 LFM3 text trial, executed only in the worker's child process."""

from collections.abc import Callable
import time

MODEL_ID = "lfm3-1.7b-bf16"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
CONTEXT_LIMIT = 262144
PREFILL_CHUNK_TOKENS = 512
GIB = 1024**3
# A provisional reservation for the full advertised window, not a measured fit.
# Short prompts also reserve it: the hub never grants an unaccounted long prefill.
REQUIREMENT = (16 * GIB, 20 * GIB)
MODEL = {
    "id": MODEL_ID,
    "repo_id": "LiquidAI/LFM3-1.7B",
    "revision": REVISION,
    "license": "lfm1.0",
    "license_url": f"https://huggingface.co/LiquidAI/LFM3-1.7B/blob/{REVISION}/LICENSE",
    "license_notice": "Experimental BF16 model and pinned checkpoint custom code. Review the LFM Open License v1.0 before downloading.",
    "estimated_bytes": 3_475_627_209,
    "kind": "llm",
    "display_name": "LFM3 1.7B BF16 (experimental)",
    "context_limit": CONTEXT_LIMIT,
    "architecture_context_limit": 262144,
}


def messages(task: dict) -> list[dict]:
    """Pass structured calls and tool-role results to the pinned LFM3 template."""
    rows = task.get("messages", [])
    if not rows:
        raise ValueError("LFM3 requires messages.")
    result = []
    for message in rows:
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported LFM3 message role.")
        content = message.get("text", message.get("content"))
        calls = message.get("tool_calls")
        if not isinstance(content, str) or (
            not content and not calls and role != "tool"
        ):
            raise ValueError("LFM3 requires nonempty text messages.")
        row = {"role": role, "content": content}
        if calls:
            if role != "assistant" or any(
                not isinstance(call.get("function", {}).get("arguments"), dict)
                for call in calls
            ):
                raise ValueError("LFM3 assistant tool arguments must be mappings.")
            row["tool_calls"] = calls
        if role == "tool":
            if (
                not isinstance(message.get("tool_call_id"), str)
                or not message["tool_call_id"]
            ):
                raise ValueError("LFM3 tool results require their call ID.")
            row["tool_call_id"] = message["tool_call_id"]
        result.append(row)
    return result


def template_inputs(tokenizer, task: dict):
    return tokenizer.apply_chat_template(
        messages(task),
        tools=task.get("tools"),
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def generate_text(model, inputs, output_tokens: int):
    """Use Transformers 5.21's hybrid-cache-aware prefill, without slicing history."""
    return model.generate(
        **inputs,
        max_new_tokens=output_tokens,
        prefill_chunk_size=PREFILL_CHUNK_TOKENS,
        cache_implementation="dynamic",
        disable_compile=True,
        use_cache=True,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.1,
        logits_to_keep=1,
    )


def reply_text(text: str) -> str:
    """Reasoning is private even if the checkpoint ignores enable_thinking=False."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        raise ValueError("LFM3 ended inside an unfinished reasoning block.")
    text = text.strip()
    while text.endswith(("<|im_end|>", "<|endoftext|>")):
        for ending in ("<|im_end|>", "<|endoftext|>"):
            text = text.removesuffix(ending).strip()
    if "<|im_start|>" in text:
        raise ValueError("LFM3 returned an unexpected extra conversation turn.")
    return text.strip()


def output_budget(prompt_tokens: int, context: int = CONTEXT_LIMIT) -> int:
    if type(context) is not int or not 1 <= context <= CONTEXT_LIMIT:
        raise ValueError("LFM3 context must be between 1 and 262144 tokens.")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("LFM3 prompt token count must be a nonnegative integer.")
    if prompt_tokens >= context:
        raise ValueError(
            "LFM3 prompt leaves no reply space in the supported context; history was not truncated."
        )
    return context - prompt_tokens


def finish_reason(generated, eos_token_id: int | list[int] | None) -> str:
    """EOS completes a turn even when it occupies the final available position."""
    endings = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id or []
    return "stop" if len(generated) and int(generated[-1]) in endings else "length"


def execute(task: dict, operation: str, model_path: str, emit: Callable) -> dict:
    if operation not in ("generate", "completion"):
        raise ValueError(
            "The LFM3 trial supports generation only; models load per request."
        )
    if task.get("model_id", task.get("model")) not in (None, MODEL_ID):
        raise ValueError("This worker serves only the explicit LFM3 BF16 checkpoint.")
    messages(task)
    context = task.get("context_limit")
    context = CONTEXT_LIMIT if context is None else context
    output_budget(0, context)

    # Inference dependencies and checkpoint code belong exclusively to the spawned
    # executor: importing them in the transport would retain CUDA allocations.
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("The LFM3 BF16 trial requires a BF16-capable CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    inputs = template_inputs(tokenizer, task)
    prompt_tokens = inputs["input_ids"].shape[-1]
    output_tokens = output_budget(prompt_tokens, context)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    inputs = inputs.to(model.device)
    torch.cuda.synchronize()
    generation_started = time.perf_counter()
    # Dense math attention can allocate [heads, chunk, full-history] scores.
    # Fail explicitly if this GPU lacks an efficient SDPA kernel for the mask.
    with (
        torch.inference_mode(),
        sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        ),
    ):
        output = generate_text(model, inputs, output_tokens)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    generated = output[0, prompt_tokens:]
    # Preserve reasoning and tool delimiters until parsing; some tokenizer builds
    # classify these as special tokens, so skipping them could leak private text.
    text = reply_text(tokenizer.decode(generated, skip_special_tokens=False))
    # Never expose an incomplete agent command: the existing runner rejects length.
    finish = finish_reason(generated, model.generation_config.eos_token_id)
    if task.get("stream") and text:
        emit({"state": "content", "content": text})
    return {
        "model": MODEL_ID,
        "text": text,
        "finish_reason": finish,
        "timing": {
            "output_tokens": len(generated),
            "prefill_tokens": prompt_tokens,
            "generation_seconds": generation_seconds,
        },
    }
