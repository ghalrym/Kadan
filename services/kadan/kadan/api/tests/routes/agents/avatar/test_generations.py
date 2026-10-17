import pytest
from pydantic import ValidationError

from kadan.api.routes.agents.avatar.generations import AvatarGenerationInput


@pytest.mark.parametrize("prompt", ["", "   ", "x" * 8001])
def test_avatar_prompt_requires_content_and_fits_hikari_limit(prompt):
    with pytest.raises(ValidationError):
        AvatarGenerationInput(prompt=prompt)


def test_generation_cannot_choose_an_external_model_or_override_image_count():
    with pytest.raises(ValidationError):
        AvatarGenerationInput(prompt="Fox", model="hosted", count=4)
