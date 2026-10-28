"""Pure formatting and validation tests; never load or run a model."""

import pytest

from experimental_worker.text import MODEL_ID, execute, messages, reply_text


def test_preserves_existing_rendered_tool_history():
    text = "<tool_response>\nSaved\n</tool_response>"
    assert messages({"messages": [{"role": "user", "text": text}]}) == [
        {"role": "user", "content": text}
    ]


@pytest.mark.parametrize(
    "rows", [[], [{"role": "tool", "text": "x"}], [{"role": "user", "text": ""}]]
)
def test_rejects_unsupported_history_without_inference(rows):
    with pytest.raises(ValueError):
        messages({"messages": rows})


def test_private_reasoning_is_removed_before_tool_parsing():
    assert (
        reply_text("<think>private</think>\n<tool_call>call</tool_call>")
        == "<tool_call>call</tool_call>"
    )


def test_unfinished_reasoning_is_an_error():
    with pytest.raises(ValueError, match="unfinished"):
        reply_text("<think>not finished")


def test_native_model_is_never_substituted():
    with pytest.raises(ValueError, match="only the explicit"):
        execute(
            {"model": "native", "messages": []}, "completion", "/unused", lambda _: None
        )


def test_manual_load_does_not_claim_residency():
    with pytest.raises(ValueError, match="generation only"):
        execute({"model": MODEL_ID}, "load", "/unused", lambda _: None)
