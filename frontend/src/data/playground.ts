// Static defaults used by the speech editor and model settings.
export const voiceDescription =
  'Warm, low female voice, mid-40s, calm and unhurried'
export const speechScript =
  'Welcome back. Your server has been up for twelve days, and everything is running normally.'
export const modelSettings = [
  {
    label: 'Language model',
    type: 'LLM',
    selected: 'Qwen3 32B Instruct',
    options: [
      'Qwen3 32B Instruct',
      'Llama 3.3 70B Instruct',
      'Mistral Small 3.1 24B',
      'Gemma 3 27B',
    ],
  },
  {
    label: 'Image generation',
    type: 'Image',
    selected: 'FLUX.1 [dev]',
    options: [
      'FLUX.1 [dev]',
      'FLUX.1 Kontext',
      'SDXL 1.0',
      'Stable Diffusion 3.5 Large',
    ],
  },
  {
    label: 'Video generation',
    type: 'Video',
    selected: 'Wan 2.2 T2V 14B',
    options: [
      'Wan 2.2 T2V 14B',
      'HunyuanVideo',
      'LTX-Video 13B',
      'CogVideoX 5B',
    ],
  },
  {
    label: 'Text to speech',
    type: 'TTS',
    selected: 'F5-TTS',
    options: ['F5-TTS', 'XTTS v2', 'Kokoro 82M', 'Parler-TTS Large'],
  },
  {
    label: 'Speech to text',
    type: 'STT',
    selected: 'Whisper Large v3 Turbo',
    options: [
      'Whisper Large v3',
      'Whisper Large v3 Turbo',
      'Whisper Medium',
      'Distil-Whisper Large v3',
    ],
  },
]
