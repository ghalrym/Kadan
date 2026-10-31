/** Hikari runs one fixed model per capability; the API owns names, sizes and licenses. */
export const capabilities = [
  { id: 'lfm3-1.7b-bf16', label: 'Language', short: 'Language', tone: 'llm' },
  { id: 'lfm3-audio-2b-bf16', label: 'Voice calls', short: 'Calls', tone: 'tts' },
  { id: 'lfm3-vl-2b-bf16', label: 'Vision', short: 'Vision', tone: 'vision' },
  { id: 'qwen-image-2.1-turbo', label: 'Image', short: 'Image', tone: 'image' },
  { id: 'h3-fl2va-int8-turbo', label: 'Video', short: 'Video', tone: 'video' },
  { id: 'qwen-tts-1.7b-custom', label: 'Text to speech', short: 'TTS', tone: 'tts' },
  { id: 'whisper-medium', label: 'Speech to text', short: 'STT', tone: 'stt' },
  { id: 'superwhisper-s1-mini', label: 'Speech formatter', short: 'Formatter', tone: 'formatter' },
  { id: 'd1-2b-cpu-f32', label: 'Decisions', short: 'Decisions', tone: 'decision' },
] as const

export type Capability = (typeof capabilities)[number]

/** Clip settings the video form starts from. */
export const videoDefaults = { duration: 5, resolution: '480p' } as const
