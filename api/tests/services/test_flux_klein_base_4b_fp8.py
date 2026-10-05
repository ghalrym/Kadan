"""Synthetic pinned download-to-native contract; no real checkpoint weights."""
import unittest

from api.tests.services.klein_quant_contract import check_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_download_selection_and_native_transformer_override(self):
        check_checkpoint(self, 'flux-klein-base-4b-fp8',
            revision='103db268c10d4d3921101b46057671f9ac460da6',
            auxiliary_revision='a3b4f4849157f664bdbc776fd7453c2783562f4d', steps=50, guidance=4.0,
            pipeline_name='Flux2KleinPipeline', quantization='fp8')
