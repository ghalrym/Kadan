"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-base-4b-nvfp4',
            revision='e655535fb8c2c148b47308cfa559a86a73273d55',
            auxiliary_revision='a3b4f4849157f664bdbc776fd7453c2783562f4d', steps=50, guidance=4.0,
            pipeline_name='Flux2KleinPipeline', quantization='nvfp4')
