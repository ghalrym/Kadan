"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-base-9b-nvfp4',
            revision='e651daf0c5d128e5cf7ffeb2da28fca22a8d7467',
            auxiliary_revision='32773329fbe7e81a90ef971740e8ba4b0364ecf3', steps=50, guidance=4.0,
            pipeline_name='Flux2KleinPipeline', quantization='nvfp4')
