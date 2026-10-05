"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-9b-fp8',
            revision='902d9d510b51533e07729f19211414a3648b77d2',
            auxiliary_revision='92196c8e11f7b6cf2b7493e037d8c5345c559216', steps=4, guidance=1.0,
            pipeline_name='Flux2KleinPipeline', quantization='fp8')
