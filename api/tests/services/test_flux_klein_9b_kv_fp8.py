"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-9b-kv-fp8',
            revision='8430c03ee457ea86907ed31a009cd56280df003d',
            auxiliary_revision='a6dfb36eca3a3906eb2fd460795adfb844e5fcce', steps=4, guidance=None,
            pipeline_name='Flux2KleinKVPipeline', quantization='fp8')
