"""Exercise native image imports and processor construction without model weights."""

from PIL import Image
from diffusers import Flux2KleinKVPipeline, Flux2KleinPipeline, Flux2Transformer2DModel, QwenImage21Pipeline
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor


def check_image_install():
    """Catch missing vision dependencies with an entirely local tiny processor."""
    tokens = ['[UNK]', '<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>']
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]')),
        unk_token='[UNK]', additional_special_tokens=tokens[1:],
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )
    result = processor(text='<|vision_start|><|image_pad|><|vision_end|>',
                       images=Image.new('RGB', (56, 56)), return_tensors='pt')
    assert result['pixel_values'].numel() > 0
    assert QwenImage21Pipeline.__module__.startswith('diffusers.pipelines.qwenimage21.')
    assert Flux2KleinPipeline.__module__.startswith('diffusers.pipelines.flux2.')
    assert Flux2KleinKVPipeline.__module__.startswith('diffusers.pipelines.flux2.')
    assert callable(Flux2Transformer2DModel.from_single_file)
    print('Native Qwen processor and Qwen/klein pipeline imports OK (no weights).')


if __name__ == '__main__':
    check_image_install()
