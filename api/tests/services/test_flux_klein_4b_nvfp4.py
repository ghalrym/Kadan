"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-4b-nvfp4',
            revision='1db2b2f776c24b76f1122e5f69ab1949fc620068',
            auxiliary_revision='e7b7dc27f91deacad38e78976d1f2b499d76a294', steps=4, guidance=1.0,
            pipeline_name='Flux2KleinPipeline', quantization='nvfp4')
