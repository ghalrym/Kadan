"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-9b-nvfp4',
            revision='e882f64f6aa086fcf8915a7763550e05af10ef13',
            auxiliary_revision='92196c8e11f7b6cf2b7493e037d8c5345c559216', steps=4, guidance=1.0,
            pipeline_name='Flux2KleinPipeline', quantization='nvfp4')
