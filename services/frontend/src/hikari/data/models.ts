/** Hikari runs one fixed model per capability; the API owns names, sizes and licenses. */
export const capabilities = [
  { id: 'inkling-2-mini', kind: 'llm', label: 'Language & Decisions', short: 'Language', tone: 'llm' },
  { id: 'flux.2-klein-4b', label: 'Image', short: 'Image', tone: 'image' },
  { id: 'qwen3-asr-1.7b', label: 'Speech to text', short: 'STT', tone: 'stt' },
  { id: 'qwen-tts-1.7b-custom', label: 'Text to speech', short: 'TTS', tone: 'tts' },
] as const

export type Capability = (typeof capabilities)[number]

/** Capabilities a subscription can serve instead of the model on this computer. */
export const subscriptionCapabilities = [
  {
    id: 'language', label: 'Language', tone: 'llm', local: 'Inkling 2 Mini on this computer',
    providers: { local: 'Local', claude: 'Claude', codex: 'Codex' },
  },
] as const
