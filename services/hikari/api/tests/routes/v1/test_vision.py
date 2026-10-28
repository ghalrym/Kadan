"""Route logic with queue boundaries replaced; no server or inference."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.routes.v1 import vision


def request(**changes):
    return vision.DescribeRequest(
        **{
            "image_base64": "YWJj",
            "media_type": "image/png",
            "question": "What is visible?",
            **changes,
        }
    )


def test_invalid_encoded_screenshot_never_enters_queue(monkeypatch):
    submit = AsyncMock()
    monkeypatch.setattr(vision.inference, "submit", submit)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            vision.describe_screenshot(request(image_base64="not base64!"), object())
        )
    assert error.value.status_code == 422
    submit.assert_not_called()


def test_valid_request_uses_vision_queue_and_validates_worker_result(monkeypatch):
    body = request()
    submit = AsyncMock(
        return_value={"model": "lfm3-vl-2b-bf16", "text": "A dialog is visible."}
    )
    monkeypatch.setattr(vision.inference, "submit", submit)

    async def await_queue(request, call):
        return await call

    monkeypatch.setattr(vision, "infer", await_queue)
    response = asyncio.run(vision.describe_screenshot(body, object()))
    submit.assert_awaited_once_with(body, feature="vision")
    assert response.text == "A dialog is visible."
    submit.return_value = {"model": "different-model", "text": "Wrong source"}
    with pytest.raises(HTTPException) as error:
        asyncio.run(vision.describe_screenshot(body, object()))
    assert error.value.status_code == 502


@pytest.mark.parametrize(
    "changes",
    [
        {"question": " "},
        {"max_new_tokens": 1025},
        {"max_new_tokens": True},
        {"image_url": "https://example.com/image.png"},
    ],
)
def test_request_rejects_unsupported_or_unbounded_inputs(changes):
    with pytest.raises(ValidationError):
        request(**changes)
