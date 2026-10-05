"""Pinned official Wan2.2-S2V-14B inference assets, excluding duplicate audio weights."""
MODEL_ID = 'wan22-s2v-14b'
REVISION = 'dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b'
REQUIRED_FILES = (
    'config.json', 'configuration.json', 'diffusion_pytorch_model.safetensors.index.json',
    'diffusion_pytorch_model-00001-of-00004.safetensors',
    'diffusion_pytorch_model-00002-of-00004.safetensors',
    'diffusion_pytorch_model-00003-of-00004.safetensors',
    'diffusion_pytorch_model-00004-of-00004.safetensors',
    'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth',
    'google/umt5-xxl/special_tokens_map.json', 'google/umt5-xxl/spiece.model',
    'google/umt5-xxl/tokenizer.json', 'google/umt5-xxl/tokenizer_config.json',
    'wav2vec2-large-xlsr-53-english/config.json',
    'wav2vec2-large-xlsr-53-english/preprocessor_config.json',
    'wav2vec2-large-xlsr-53-english/special_tokens_map.json',
    'wav2vec2-large-xlsr-53-english/vocab.json',
    'wav2vec2-large-xlsr-53-english/model.safetensors',
)
