import pytest
from pydantic import ValidationError

from kadan.api.routes.agents.avatar import AvatarInput


def test_remove_avatar_requires_explicit_null():
    assert AvatarInput(asset_id=None).asset_id is None
    with pytest.raises(ValidationError):
        AvatarInput()


def test_avatar_cannot_reference_an_external_image_url():
    with pytest.raises(ValidationError):
        AvatarInput(asset_id="https://example.com/avatar.png")
