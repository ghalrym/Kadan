"""Exercise the shared bridge with synthetic checkpoints, without enabling catalog models."""
import unittest
from unittest.mock import patch

from api.inference.flux_klein import RECIPES
from api.inference.native_image import ImageRecipe
from api.services.model_catalog import CATALOG, CatalogEntry, CheckpointSource
from api.tests.services.klein_quant_contract import check_checkpoint


class QuantizedCheckpointContractTests(unittest.TestCase):
    def test_download_selection_decode_and_publication_for_both_formats(self):
        for quantization in ('fp8', 'nvfp4'):
            with self.subTest(quantization=quantization):
                model_id = 'synthetic-klein'
                entry = CatalogEntry(model_id, 'fixture/quant', 'a' * 40, 'apache-2.0', 0,
                    kind='image', layout='components', source_files=('quant.safetensors',),
                    required_files=('model_index.json', 'transformer/config.json',
                        'tokenizer/tokenizer.json', 'scheduler/scheduler_config.json',
                        'text_encoder/config.json', 'vae/config.json'),
                    weight_paths=('', 'text_encoder', 'vae'),
                    auxiliary_sources=(CheckpointSource('fixture/base', 'b' * 40,
                        files=('model_index.json', 'transformer/config.json'),
                        component_paths=('tokenizer', 'scheduler', 'text_encoder', 'vae')),))
                recipe = ImageRecipe(model_id, entry.revision, 'Flux2KleinPipeline',
                    {'1:1': (16, 16)}, 4, 1., quantization, 'quant.safetensors')
                with patch.dict(CATALOG, {model_id: entry}), patch.dict(RECIPES, {model_id: recipe}):
                    check_checkpoint(self, model_id, revision=entry.revision,
                        auxiliary_revision='b' * 40, steps=4, guidance=1.,
                        pipeline_name=recipe.pipeline, quantization=quantization)
