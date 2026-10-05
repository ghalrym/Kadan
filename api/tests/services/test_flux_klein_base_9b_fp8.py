"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-base-9b-fp8',
            revision='9ecf2143d71542449960c5584340269c6d401449',
            auxiliary_revision='32773329fbe7e81a90ef971740e8ba4b0364ecf3', steps=50, guidance=4.0,
            pipeline_name='Flux2KleinPipeline', quantization='fp8')
