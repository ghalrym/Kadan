"""Verified FLUX.2 klein checkpoint recipes; no implicit weight downloads."""
from api.inference.native_image import ImageRecipe

SIZES = {'1:1': (1024, 1024), '4:3': (1152, 864), '3:4': (864, 1152), '16:9': (1344, 768)}
RECIPES = {
    'flux-klein-4b': ImageRecipe('flux-klein-4b', 'e7b7dc27f91deacad38e78976d1f2b499d76a294',
        'Flux2KleinPipeline', SIZES, 4, 1.0),
}

RECIPES['flux-klein-9b-fp8'] = ImageRecipe('flux-klein-9b-fp8',
    '902d9d510b51533e07729f19211414a3648b77d2', 'Flux2KleinPipeline', SIZES, 4, 1.0,
    quantization='fp8', weight_filename='flux-2-klein-9b-fp8.safetensors')
