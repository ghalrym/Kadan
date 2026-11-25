from uuid import UUID, uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.api.routes.agents.space.slides import get_slides

DECK = {
    "schema_version": 1,
    "width": 1920,
    "height": 1080,
    "slides": [
        {
            "id": str(uuid4()),
            "layout": "title",
            "title": "Hello",
            "body": "World",
            "notes": "Say hi",
            "image_asset_id": None,
        }
    ],
}


def test_slides_pass_through_the_deck(tsuchi, call):
    deck = tsuchi.add("deck", "Pitch", deck=DECK)
    slides = call(get_slides, "alpha", UUID(deck["id"]))
    assert (slides.type, slides.title) == ("slides", "Pitch")
    assert [slide.model_dump(mode="json") for slide in slides.slides] == DECK["slides"]


def test_pages_are_not_slides(tsuchi, call):
    page = tsuchi.add("space", "Page")
    with pytest.raises(ServiceError, match="This item no longer exists"):
        call(get_slides, "alpha", UUID(page["id"]))
