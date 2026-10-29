"""Bounded screenshot understanding with the released BF16 LFM3-VL checkpoint."""

import base64
import binascii
import io
import json
import re
from collections.abc import Callable
from pathlib import Path

MODEL = json.loads(Path(__file__).with_name("catalog_vision.json").read_text())
# Provisional conservative admission bounds; physical fit still requires a trial.
REQUIREMENT = (8 * 1024**3, 16 * 1024**3)
MAX_IMAGE_BYTES = 2 * 1024**2
MAX_PIXELS = 4_194_304
MAX_CONTEXT = 128000


def validate_context(prompt_tokens: int, output_tokens: int) -> None:
    """Keep the full prompt and reserved reply inside the checkpoint's limit."""
    if prompt_tokens < 1 or output_tokens < 1:
        raise ValueError("Screenshot input and output must contain tokens.")
    if prompt_tokens + output_tokens > MAX_CONTEXT:
        raise ValueError(
            "Screenshot, question and reserved reply exceed the model's "
            "128000-token context; no input was truncated."
        )


def output_budget(prompt_tokens: int, requested_tokens: int | None = None) -> int:
    if requested_tokens is not None and (
        type(requested_tokens) is not int or requested_tokens < 1
    ):
        raise ValueError("A screenshot response needs a positive integer output budget")
    if prompt_tokens >= MAX_CONTEXT:
        raise ValueError(
            "Screenshot and question leave no output space in the 128000-token context; no input was truncated."
        )
    budget = (
        MAX_CONTEXT - prompt_tokens if requested_tokens is None else requested_tokens
    )
    validate_context(prompt_tokens, budget)
    return budget


def validate_completion(count: int, last_token: int | None, eos_ids) -> None:
    endings = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    if count < 1 or last_token not in endings:
        raise ValueError(
            "The vision model ended without a terminal EOS; the observation is incomplete."
        )


def description(text: str) -> str:
    """Return the final observation, rejecting an unfinished reasoning block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # A generation prompt may already contain the opening think token.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    while text.endswith(("<|im_end|>", "<|endoftext|>")):
        for ending in ("<|im_end|>", "<|endoftext|>"):
            if text.endswith(ending):
                text = text.removesuffix(ending).strip()
    if "<think>" in text or not text:
        raise ValueError("The vision model did not complete a screenshot description.")
    if "<|im_start|>" in text:
        raise ValueError("The vision model generated another conversation turn.")
    return text


def image_bytes(task: dict) -> bytes:
    if task.get("media_type") not in ("image/png", "image/jpeg"):
        raise ValueError("Screenshot must be PNG or JPEG.")
    encoded = task.get("image_base64", "")
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Screenshot exceeds the 2 MiB compressed limit.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Screenshot must contain valid base64 image bytes.") from error
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Screenshot must contain at most 2 MiB of image bytes.")
    signature = (
        b"\x89PNG\r\n\x1a\n" if task["media_type"] == "image/png" else b"\xff\xd8\xff"
    )
    if not content.startswith(signature):
        raise ValueError("Screenshot signature does not match its declared media type.")
    return content


def execute(
    task: dict,
    operation: str,
    model_path: str,
    emit: Callable[[dict], None],
) -> dict:
    if operation != "generate":
        raise ValueError(
            "The vision worker supports screenshot generation requests only."
        )
    if task.get("model", MODEL["id"]) != MODEL["id"]:
        raise ValueError("This worker serves only the pinned LFM3-VL BF16 model.")
    question = task.get("question", "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A screenshot question cannot be blank.")
    tokens = task.get("max_new_tokens")
    if tokens is not None and (
        type(tokens) is not int or not 1 <= tokens < MAX_CONTEXT
    ):
        raise ValueError(
            "A screenshot response requires an output budget below 128000 tokens."
        )
    content = image_bytes(task)

    # Heavy inference dependencies load only inside the isolated job process.
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("LFM3-VL BF16 requires a CUDA device with BF16 support.")
    with Image.open(io.BytesIO(content)) as source:
        expected = "PNG" if task["media_type"] == "image/png" else "JPEG"
        if source.format != expected or source.width * source.height > MAX_PIXELS:
            raise ValueError(
                "Screenshot format must match its media type and fit 4,194,304 pixels."
            )
        image = source.convert("RGB")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    tokens = output_budget(prompt_tokens, tokens)
    emit({"state": "running", "model": MODEL["id"]})
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            max_new_tokens=tokens,
            logits_to_keep=1,
        )
    generated = outputs[:, prompt_tokens:]
    validate_completion(
        generated.shape[1],
        int(generated[0, -1]) if generated.shape[1] else None,
        model.config.text_config.eos_token_id,
    )
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    return {"model": MODEL["id"], "text": description(text)}
