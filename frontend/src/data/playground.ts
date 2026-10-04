// Fixed design fixtures, not server responses. Nothing in this module changes at runtime.
export const chatMessages = [
  {
    role: 'user',
    text: "We saw p95 latency jump from 1.1s to 4.8s around 14:02. The request log shows a burst of image edits from batch-worker. What's the likely cause?",
  },
  {
    role: 'assistant',
    text: "Most likely GPU contention. Image edits run on the same device as the LLM, and a burst of them holds the GPU long enough to queue chat requests behind them.\n\nThree things to check:\n1. Queue time vs. inference time on the slow chat requests. If queue time dominates, it's contention, not the model.\n2. Whether batch-worker sets a concurrency limit. Without one it can take every slot.\n3. VRAM headroom. If the image model pushed the KV cache out, the LLM will re-prefill long prompts.\n\nA per-client rate limit on /v1/images/edits would stop it happening again.",
    meta: '214 tokens · 3.1 s',
  },
]
export const decisionState =
  'Image generation has returned HTTP 500 for the last hour. Logs show "CUDA out of memory while allocating 2.1 GiB" on GPU 0. Batch jobs from batch-worker are stuck and customers are waiting on renders.'
export type DecisionQuestion = { key: string; instructions: string } & (
  | { type: 'Choice'; options: { key: string; description: string }[] }
  | { type: 'Score'; levels: string[] }
  | { type: 'Noul'; trueWhen: string; falseWhen: string }
)
export const questions: DecisionQuestion[] = [
  {
    key: 'cause',
    type: 'Choice',
    instructions: 'What is most likely causing the failures',
    options: [
      { key: 'gpu_capacity', description: 'Out of memory or GPU saturation' },
      {
        key: 'model_bug',
        description: 'Model returns wrong or malformed output',
      },
      { key: 'client_error', description: 'Bad requests from the caller' },
      { key: 'network', description: 'Timeouts or connectivity problems' },
    ],
  },
  {
    key: 'severity',
    type: 'Score',
    instructions: 'How severe the impact on users is',
    levels: ['Minor inconvenience', 'Degraded service', 'Full outage'],
  },
  {
    key: 'is_urgent',
    type: 'Noul',
    instructions: 'The message conveys urgency or time-sensitivity',
    trueWhen: '',
    falseWhen: '',
  },
]
export const imageSets = [
  {
    id: 'image-set-2',
    mode: 'Generate',
    prompt:
      'isometric server room at dawn, soft volumetric light, muted palette',
    aspect: 'landscape' as const,
    seeds: [48213, 48214, 48215, 48216],
    meta: 'FLUX.1 [dev] · 1152×864 · 14:21',
  },
  {
    id: 'image-set-1',
    mode: 'Edit',
    prompt:
      'replace the sky with a stormy overcast, keep the building untouched',
    aspect: 'wide' as const,
    seeds: [90377, 90378],
    meta: 'FLUX.1 Kontext · 1344×768 · 14:08',
  },
]
export interface VideoJob {
  id: string
  prompt: string
  duration: string
  resolution: string
  aspect: 'wide' | 'portrait'
  fps: string
  progress: number
  time: string
  status: 'Rendering' | 'Queued' | 'Done'
  thumbnail: string
  progressText: string
}
export const videoJobs: VideoJob[] = [
  {
    id: 'vid_3f9a21',
    prompt:
      'Slow dolly across a rain-soaked neon street at night, reflections on the asphalt, cinematic',
    duration: '8s',
    resolution: '720p',
    aspect: 'wide',
    fps: '24',
    progress: 38,
    time: '14:26',
    status: 'Rendering',
    thumbnail: 'Frame 72 / 192',
    progressText: '38% · ~0:14 left',
  },
  {
    id: 'vid_77c0e4',
    prompt:
      'Macro shot of coffee being poured into a glass cup, steam rising, morning light',
    duration: '4s',
    resolution: '1080p',
    aspect: 'portrait',
    fps: '30',
    progress: 0,
    time: '14:27',
    status: 'Queued',
    thumbnail: 'In queue',
    progressText: 'waiting',
  },
  {
    id: 'vid_1b52d8',
    prompt: 'Drone shot rising over a foggy pine forest at sunrise',
    duration: '12s',
    resolution: '720p',
    aspect: 'wide',
    fps: '24',
    progress: 100,
    time: '13:52',
    status: 'Done',
    thumbnail: '12s · 720p',
    progressText: '12s rendered',
  },
]
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
