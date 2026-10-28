"""Pure input/metadata tests; no model import, loading or inference."""

import base64

import pytest

from api.pydantic_models.inference_task import InferenceTask
from experimental_worker.vision import (
    MAX_CONTEXT,
    MODEL,
    description,
    execute,
    image_bytes,
    validate_context,
)


def test_context_uses_exact_checkpoint_maximum_and_reserves_output_without_truncation():
    assert (
        MAX_CONTEXT
        == MODEL["context_limit"]
        == MODEL["architecture_context_limit"]
        == 128000
    )
    validate_context(8000, 1024)
    validate_context(126976, 1024)
    with pytest.raises(ValueError, match="no input was truncated"):
        validate_context(126977, 1024)


@pytest.mark.parametrize("prompt, output", [(0, 1024), (1, 0), (-1, 1024)])
def test_context_requires_positive_prompt_and_reply_tokens(prompt, output):
    with pytest.raises(ValueError, match="must contain tokens"):
        validate_context(prompt, output)


def test_binary_screenshot_is_recovered_without_interpreting_paths_or_urls():
    assert image_bytes({"media_type": "image/png", "image_base64": "YWJj"}) == b"abc"
    for value in ("https://example.com/screenshot.png", "/tmp/screenshot.png", ""):
        with pytest.raises(ValueError):
            image_bytes({"media_type": "image/png", "image_base64": value})


def test_screenshot_payload_is_bounded_and_supported_media_only():
    for task in (
        {"media_type": "image/gif", "image_base64": "YWJj"},
        {"media_type": "image/png", "image_base64": "A" * 2_796_205},
        {
            "media_type": "image/jpeg",
            "image_base64": base64.b64encode(b"a" * (2 * 1024**2 + 1)).decode(),
        },
    ):
        with pytest.raises(ValueError):
            image_bytes(task)


def test_vision_is_its_own_task_and_cannot_replace_image_generation():
    task = InferenceTask(
        input={
            "feature": "vision",
            "media_type": "image/png",
            "image_base64": "YWJj",
            "question": "What is visible?",
        }
    )
    assert task.feature == "vision"
    assert task.input.model == "lfm3-vl-2b-bf16"
    assert InferenceTask.model_validate_json(task.model_dump_json()) == task


@pytest.mark.parametrize(
    "changes",
    [
        {"question": " "},
        {"max_new_tokens": 1025},
        {"max_new_tokens": True},
        {"model": "another-model"},
    ],
)
def test_invalid_work_is_rejected_before_loading_runtime_dependencies(changes):
    with pytest.raises(ValueError):
        execute(
            {
                "question": "Describe",
                "media_type": "image/png",
                "image_base64": "YWJj",
                **changes,
            },
            "generate",
            "/unused",
            lambda event: None,
        )


def test_final_description_omits_reasoning_and_requires_a_completed_answer():
    assert description("<think>internal</think>Visible button") == "Visible button"
    assert description("internal</think>Visible button<|im_end|>") == "Visible button"
    assert description("Visible button<|im_end|><|endoftext|>") == "Visible button"
    for text in ("", "<think>unfinished", "<think>only reasoning</think>"):
        with pytest.raises(ValueError):
            description(text)
    with pytest.raises(ValueError, match="another conversation turn"):
        description("Visible button<|im_start|>user")


def test_pinned_manifest_has_weights_custom_code_and_license():
    assert len(MODEL["revision"]) == 40
    assert MODEL["kind"] == "vision"
    files = {item["name"]: item for item in MODEL["artifacts"]}
    assert {
        "LICENSE",
        "model.safetensors",
        "modeling_lfm3_vl.py",
        "processing_lfm3_vl.py",
    } <= files.keys()
    assert MODEL["estimated_bytes"] == sum(item["size"] for item in files.values())
    assert files["model.safetensors"]["algorithm"] == "sha256"
    assert all(MODEL["revision"] in item["url"] for item in files.values())
