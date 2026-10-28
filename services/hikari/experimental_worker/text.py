"""Bounded BF16 LFM3 text trial, executed only in the worker's child process."""

from collections.abc import Callable

MODEL_ID = "lfm3-1.7b-bf16"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
CONTEXT_LIMIT = 16384
MAX_NEW_TOKENS = 1024
GIB = 1024**3
REQUIREMENT = (8 * GIB, 8 * GIB)
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


def messages(task: dict) -> list[dict[str, str]]:
    """Keep the existing Kadan text/tool rendering and its authority boundary."""
    rows = task.get("messages", [])
    if not rows:
        raise ValueError("LFM3 requires messages.")
    result = []
    for message in rows:
        if message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("Unsupported LFM3 message role.")
        content = message.get("text", message.get("content"))
        if not isinstance(content, str) or not content:
            raise ValueError("LFM3 requires nonempty text messages.")
        result.append({"role": message["role"], "content": content})
    if sum(len(row["content"]) for row in result) > 128_000:
        raise ValueError("LFM3 text exceeds the experimental input limit.")
    return result


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


def execute(task: dict, operation: str, model_path: str, emit: Callable) -> dict:
    if operation not in ("generate", "completion"):
        raise ValueError(
            "The LFM3 trial supports generation only; models load per request."
        )
    if task.get("model_id", task.get("model")) not in (None, MODEL_ID):
        raise ValueError("This worker serves only the explicit LFM3 BF16 checkpoint.")
    conversation = messages(task)
    context = task.get("context_limit")
    context = CONTEXT_LIMIT if context is None else context
    if type(context) is not int or not MAX_NEW_TOKENS < context <= CONTEXT_LIMIT:
        raise ValueError("LFM3 trial context must be between 1025 and 16384 tokens.")

    # Inference dependencies and checkpoint code belong exclusively to the spawned
    # executor: importing them in the transport would retain CUDA allocations.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("The LFM3 BF16 trial requires a BF16-capable CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_tokens = inputs["input_ids"].shape[-1]
    if prompt_tokens + MAX_NEW_TOKENS > context:
        raise ValueError(
            "LFM3 prompt and reserved reply exceed the trial context; history was not truncated."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.1,
        )
    generated = output[0, prompt_tokens:]
    # Preserve reasoning and tool delimiters until parsing; some tokenizer builds
    # classify these as special tokens, so skipping them could leak private text.
    text = reply_text(tokenizer.decode(generated, skip_special_tokens=False))
    # Never expose an incomplete agent command: the existing runner rejects length.
    finish = "length" if len(generated) >= MAX_NEW_TOKENS else "stop"
    if task.get("stream") and text:
        emit({"state": "content", "content": text})
    return {"model": MODEL_ID, "text": text, "finish_reason": finish}
