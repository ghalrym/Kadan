import type { TierId, TierModel } from '../api/settings'

/** Presentation for the API's hardware tiers; the API owns names, sizes and models. */
export const tierCopy: Record<TierId, { tagline: string; recommended: boolean }> = {
  go: { tagline: 'Laptops and entry-level GPUs', recommended: false },
  plus: { tagline: 'Mid-range gaming GPUs', recommended: false },
  pro: { tagline: 'Single-GPU workstations', recommended: true },
  ultra: { tagline: 'High-end workstations and servers', recommended: false },
}

export const modalityLabels: Record<TierModel['modality'], { label: string; short: string }> = {
  llm: { label: 'Language', short: 'Language' },
  image: { label: 'Image', short: 'Image' },
  video: { label: 'Video', short: 'Video' },
  speech: { label: 'Text to speech', short: 'TTS' },
  transcription: { label: 'Speech to text', short: 'STT' },
  formatting: { label: 'Speech formatter', short: 'Formatter' },
}
