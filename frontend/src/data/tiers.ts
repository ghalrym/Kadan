// Fixed design fixtures until tier settings are available through the API.
export type TierId = 'tier-1' | 'tier-2' | 'tier-3' | 'tier-4'

export interface Tier {
  id: TierId
  name: string
  vramGb: number
  ramGb: number
  tagline: string
  recommended: boolean
  models: Record<ModelCategory, string>
}

// TODO: Replace placeholder model names once each tier's models are chosen.
const placeholderModels: Record<ModelCategory, string> = {
  language: 'TBD',
  image: 'TBD',
  video: 'TBD',
  tts: 'TBD',
  stt: 'TBD',
  'speech-formatter': 'Whisper: S1-mini',
}

export const tiers: Tier[] = [
  { id: 'tier-1', name: 'Go', vramGb: 8, ramGb: 32, tagline: 'Laptops and entry-level GPUs', recommended: false, models: placeholderModels },
  { id: 'tier-2', name: 'Plus', vramGb: 16, ramGb: 64, tagline: 'Mid-range gaming GPUs', recommended: false, models: placeholderModels },
  { id: 'tier-3', name: 'Pro', vramGb: 24, ramGb: 128, tagline: 'Single-GPU workstations', recommended: true, models: placeholderModels },
  { id: 'tier-4', name: 'Ultra', vramGb: 48, ramGb: 512, tagline: 'High-end workstations and servers', recommended: false, models: placeholderModels },
]

export type ModelCategory =
  | 'language'
  | 'image'
  | 'video'
  | 'tts'
  | 'stt'
  | 'speech-formatter'

/** Every tier includes one model per category. */
export const modelCategories: {
  id: ModelCategory
  label: string
  shortLabel: string
}[] = [
  { id: 'language', label: 'Language', shortLabel: 'Language' },
  { id: 'image', label: 'Image', shortLabel: 'Image' },
  { id: 'video', label: 'Video', shortLabel: 'Video' },
  { id: 'tts', label: 'Text to speech', shortLabel: 'TTS' },
  { id: 'stt', label: 'Speech to text', shortLabel: 'STT' },
  { id: 'speech-formatter', label: 'Speech formatter', shortLabel: 'Formatter' },
]

export interface ModelDownloadPreview {
  category: ModelCategory
  status: 'queued' | 'downloading' | 'complete'
  percent: number
}

export const downloadPreview: ModelDownloadPreview[] = [
  { category: 'language', status: 'complete', percent: 100 },
  { category: 'image', status: 'downloading', percent: 40 },
  { category: 'video', status: 'queued', percent: 0 },
  { category: 'tts', status: 'queued', percent: 0 },
  { category: 'stt', status: 'queued', percent: 0 },
  { category: 'speech-formatter', status: 'queued', percent: 0 },
]
