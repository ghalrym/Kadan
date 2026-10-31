import unittest
from unittest.mock import AsyncMock, patch

from api.pydantic_models.model_store import ModelStoreStatus
from api.routes.v1 import models


class ModelListingTests(unittest.TestCase):
    def test_listing_reports_fixed_language_assignment_without_loading(self):
        snapshot = ModelStoreStatus(models=[])
        with (
            patch.object(models.inference, "models_status", return_value=snapshot),
            patch.object(models.inference, "submit", new=AsyncMock()) as submit,
        ):
            result = models.list_models()
        self.assertEqual(result.selected_model_id, "lfm3-1.7b-bf16")
        submit.assert_not_awaited()
