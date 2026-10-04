import unittest

from pydantic import ValidationError

from api.routes.v1.images.edits import ImageRequest


class ImageEditValidationTests(unittest.TestCase):
    def test_image_edit_requires_source(self):
        with self.assertRaises(ValidationError):
            ImageRequest(prompt="Change the sky")

    def test_image_strength_must_be_in_range(self):
        with self.assertRaises(ValidationError):
            ImageRequest(prompt="Change the sky", image="source.png", strength=1.1)
