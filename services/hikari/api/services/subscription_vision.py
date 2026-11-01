"""One screenshot observation using the existing reusable subscription sign-in."""

import io
import json
from pathlib import Path

from api.pydantic_models.subscriptions import SUBSCRIPTION_MODELS, ActiveSubscription
from api.services.inference_errors import InferenceFailure
from api.services.subscription_inference import SubscriptionStream
from experimental_worker.vision import MAX_PIXELS, image_bytes
from PIL import Image, UnidentifiedImageError


class SubscriptionVision(SubscriptionStream):
    def __init__(self, subscription: ActiveSubscription, body):
        self.subscription = subscription
        self.model = SUBSCRIPTION_MODELS[subscription.provider]
        self.system = "Describe the supplied image in answer to the user's question. Return only the observation. Do not use tools."
        self.prompt = body.question
        self.body = body
        self.content = image_bytes(body.model_dump())
        try:
            with Image.open(io.BytesIO(self.content)) as image:
                expected = "PNG" if body.media_type == "image/png" else "JPEG"
                if image.format != expected or image.width * image.height > MAX_PIXELS:
                    raise ValueError(
                        "Screenshot format must match its media type and fit 4,194,304 pixels."
                    )
                image.verify()
        except (
            ValueError,
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as error:
            raise InferenceFailure(
                "Screenshot must be a valid PNG or JPEG and fit 4,194,304 pixels.", 422
            ) from error
        if body.max_new_tokens is not None:
            raise InferenceFailure(
                "Subscription vision does not support a reply token budget. Remove the budget or select Local.",
                422,
            )
        self.home: Path | None = None
        self.process = None
        self.stderr = None

    def _command(self):
        command, env = super()._command()
        if self.subscription.provider == "claude":
            command += ["--input-format", "stream-json"]
            self.prompt = (
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": self.body.media_type,
                                        "data": self.body.image_base64,
                                    },
                                },
                                {"type": "text", "text": self.body.question},
                            ],
                        },
                        "parent_tool_use_id": None,
                    }
                )
                + "\n"
            )
        else:
            image = self.home / (
                "screenshot.png"
                if self.body.media_type == "image/png"
                else "screenshot.jpg"
            )
            image.write_bytes(self.content)
            command[-1:-1] = ["--image", str(image)]
            self.prompt = self.system + "\n\n" + self.body.question
        return command, env
