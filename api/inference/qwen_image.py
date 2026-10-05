"""Pinned Qwen-Image-2.1 recipe for native generation and editing."""
from api.inference.native_image import ImageRecipe, generate as generate_native, native_modules

MODEL_ID = 'qwen-image-2.1'
REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
SIZES = {'1:1': (2048, 2048), '4:3': (2400, 1792), '3:4': (1792, 2400), '16:9': (2752, 1536)}
RECIPE = ImageRecipe(MODEL_ID, REVISION, 'QwenImage21Pipeline', SIZES, 40)


def generate(path, resources, prompt, aspect, seeds, cancel, device='cuda:0', image=None, modules=native_modules):
    """Run the verified checkpoint recipe under the shared native lifecycle."""
    return generate_native(path, resources, prompt, aspect, seeds, cancel, device, image, modules, recipe=RECIPE)
