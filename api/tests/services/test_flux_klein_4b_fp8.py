"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-4b-fp8',
            revision='5b4408e59397a4a37ccb46afe426d8ed86379441',
            auxiliary_revision='e7b7dc27f91deacad38e78976d1f2b499d76a294', steps=4, guidance=1.0,
            pipeline_name='Flux2KleinPipeline', quantization='fp8')
