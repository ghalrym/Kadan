/** Hikari runs one fixed model per capability; the API owns names, sizes and licenses. */
export const capabilities = [
  { id: 'inkling-2-mini', kind: 'llm', label: 'Language & Decisions', short: 'Language', tone: 'llm' },
  { id: 'lfm3-audio-2b-bf16', label: 'Voice calls', short: 'Calls', tone: 'tts' },
  { id: 'qwen-image-2.1-turbo', label: 'Image', short: 'Image', tone: 'image' },
  { id: 'whisper-medium', label: 'Speech to text', short: 'STT', tone: 'stt' },
  { id: 'superwhisper-s1-mini', label: 'Speech formatter', short: 'Formatter', tone: 'formatter' },
  { id: 'qwen-tts-1.7b-custom', label: 'Text to speech', short: 'TTS', tone: 'tts' },
  { id: 'h3-fl2va-int8-turbo', label: 'Video', short: 'Video', tone: 'video' },
] as const

export type Capability = (typeof capabilities)[number]

/** Clip settings the video form starts from. */
export const videoDefaults = { duration: 5, resolution: '480p' } as const

/** Capabilities a subscription can serve instead of the model on this computer. */
export const subscriptionCapabilities = [
  {
    id: 'language', label: 'Language', tone: 'llm', local: 'Inkling 2 Mini on this computer',
    providers: { local: 'Local', claude: 'Claude', codex: 'Codex' },
  },
] as const
