"""Verified FLUX.2 klein checkpoint recipes; no implicit weight downloads."""
from api.inference.native_image import ImageRecipe

SIZES = {'1:1': (1024, 1024), '4:3': (1152, 864), '3:4': (864, 1152), '16:9': (1344, 768)}
RECIPES = {
    'flux-klein-4b': ImageRecipe('flux-klein-4b', 'e7b7dc27f91deacad38e78976d1f2b499d76a294',
        'Flux2KleinPipeline', SIZES, 4, 1.0),
}

RECIPES['flux-klein-4b-nvfp4'] = ImageRecipe('flux-klein-4b-nvfp4',
    '1db2b2f776c24b76f1122e5f69ab1949fc620068', 'Flux2KleinPipeline', SIZES, 4, 1.0,
    quantization='nvfp4', weight_filename='flux-2-klein-4b-nvfp4.safetensors')
